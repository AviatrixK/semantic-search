# Deploying to one EC2 instance (ap-south-1)

Target: Ubuntu 24.04, t3.medium (2 vCPU, 4 GB), user `ubuntu`, Docker Compose, Caddy with an automatic certificate for an
`sslip.io` hostname, videos in a private S3 bucket reached through the instance's IAM role.

```
browser --https--> caddy :443 -+- /api/*, /auth/* --> api (gunicorn + uvicorn) --> postgres, redis, S3
                               +- everything else --> built React app (index.html fallback)
                 worker (Celery: ffmpeg + Whisper + embeddings) and beat (hourly token purge) share postgres/redis/S3
browser --https--> <bucket>.s3.ap-south-1.amazonaws.com   (pre-signed GET URLs, <video> tag only)
```

Only Caddy publishes ports (80, 443). Postgres and Redis are reachable only from other containers.

Commands marked **laptop** run in PowerShell on your PC (AWS CLI v2 configured with an admin-capable profile); **server**
commands run over SSH as `ubuntu`. Replace `YOUR-BUCKET` with a globally unique name, e.g. `svs-videos-yourname-aps1`, and
`13.201.45.67` / `13-201-45-67.sslip.io` with your Elastic IP / hostname.

---

## 1. First deploy

### 1.1 S3 bucket (laptop)

```powershell
$B = "YOUR-BUCKET"; $R = "ap-south-1"
aws s3api create-bucket --bucket $B --region $R --create-bucket-configuration LocationConstraint=$R
aws s3api put-public-access-block --bucket $B --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

The bucket stays private (new buckets are encrypted with SSE-S3 by default). No CORS rule is needed: the frontend only loads S3
objects through `<video>` elements (no `fetch`/XHR, no `crossOrigin` attribute, no canvas). If you ever add one of those, the
bucket needs a CORS rule allowing `GET`/`HEAD` from `https://<DOMAIN>`.

### 1.2 IAM role for the instance (laptop)

```powershell
$B = "YOUR-BUCKET"
@'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
'@ | Set-Content -Encoding ascii trust.json
@"
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":"s3:ListBucket","Resource":"arn:aws:s3:::$B"},
 {"Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:DeleteObject"],"Resource":"arn:aws:s3:::$B/*"}]}
"@ | Set-Content -Encoding ascii policy.json
aws iam create-role --role-name svs-ec2 --assume-role-policy-document file://trust.json
aws iam put-role-policy --role-name svs-ec2 --policy-name svs-s3 --policy-document file://policy.json
aws iam create-instance-profile --instance-profile-name svs-ec2
aws iam add-role-to-instance-profile --instance-profile-name svs-ec2 --role-name svs-ec2
Remove-Item trust.json, policy.json
```

`s3:ListBucket` lets the app's startup check (`HeadBucket`) pass and makes a missing object a 404 instead of a 403.

### 1.3 Launch the instance (console or CLI)

* AMI: Ubuntu Server 24.04 LTS, type **t3.medium**, region **ap-south-1**, **40 GB gp3** root volume.
* IAM instance profile: `svs-ec2`.
* Security group inbound: **22** from *your IP only*, **80** and **443** from anywhere. Nothing else (not 5432, 6379, 8000).
* Allocate an **Elastic IP** and associate it (otherwise the IP, and so the hostname, changes on every stop/start).
* **Containers must be able to read the instance role.** IMDSv2 defaults to a hop limit of 1, which blocks containers
  ("Unable to locate credentials"). Set it to 2 (laptop):

```powershell
aws ec2 modify-instance-metadata-options --region ap-south-1 --instance-id i-0123456789abcdef0 --http-tokens required --http-put-response-hop-limit 2 --http-endpoint enabled
```

Hostname: Elastic IP `13.201.45.67` becomes `13-201-45-67.sslip.io` (dots to dashes). Check from the laptop:
`nslookup 13-201-45-67.sslip.io` must return your IP.

### 1.4 Prepare the server (server)

```bash
ssh -i ~/.ssh/YOUR-KEY.pem ubuntu@13.201.45.67

sudo apt-get update && sudo apt-get install -y ca-certificates curl git
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
exit
```

Log in again so the `docker` group applies, then:

```bash
ssh -i ~/.ssh/YOUR-KEY.pem ubuntu@13.201.45.67
docker compose version

# 2 GB swap: a safety net while Whisper and the API both hold models in 4 GB of RAM
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 1.5 Get the code and configure it (server)

```bash
git clone https://github.com/AviatrixK/semantic-search.git ~/svs      # private repo: use a deploy key or a token
cd ~/svs
cp .env.prod.example .env.prod && chmod 600 .env.prod

# generate secrets and put them in
PW=$(openssl rand -hex 24); JWT=$(openssl rand -base64 48 | tr -d '\n=')
sed -i "s|REPLACE_WITH_openssl_rand_hex_24|$PW|g; s|REPLACE_WITH_A_LONG_RANDOM_STRING|$JWT|" .env.prod
nano .env.prod       # set DOMAIN, S3_BUCKET, GEMINI_API_KEY; check AWS_REGION=ap-south-1
```

The app refuses to start if `JWT_SECRET` or the database password is a known default, or if `COOKIE_SECURE` is not `true`.
`POSTGRES_PASSWORD` is only read when the database volume is first created.

### 1.6 Build and start (server)

```bash
./scripts/deploy.sh --no-pull
```

The first build downloads CPU-only PyTorch and bakes the Whisper and embedding models into the image: expect 10-20 minutes,
much less when layers are cached. The script waits for the health checks and prints `docker compose ps`. Postgres creates the schema
from `db/init.sql` on the very first start (empty volume only).

Short alias used below:

```bash
alias dc='docker compose --env-file .env.prod -f docker-compose.prod.yml'
```

---

## 2. Create the first admin (server)

```bash
cd ~/svs
 dc exec api python -m app.scripts.create_admin you@example.com 'A-strong-passw0rd'
```

Passwords need 8+ characters with a letter and a digit. The leading space keeps the command out of shell history on Ubuntu.

---

## 3. Verify

```bash
dc ps                                                                              # every service "healthy"
curl -sI https://13-201-45-67.sslip.io/ | head -n 5                                # 200, valid certificate (no -k)
curl -s -o /dev/null -w '%{http_code}\n' https://13-201-45-67.sslip.io/api/videos  # 401: API reachable, auth enforced
curl -s -o /dev/null -w '%{http_code}\n' https://13-201-45-67.sslip.io/some/route  # 200: index.html fallback
dc exec api python -c "import boto3,os; print(boto3.client('s3',region_name='ap-south-1').head_bucket(Bucket=os.environ['S3_BUCKET'])['ResponseMetadata']['HTTPStatusCode'])"   # 200
```

From your laptop, the database and Redis must be unreachable: `Test-NetConnection 13.201.45.67 -Port 5432` and `-Port 6379`
must both report `TcpTestSucceeded : False`.

In the browser (`https://13-201-45-67.sslip.io`): log in, upload a short video on `/admin`, watch the job reach *done*, search,
open a result (the player must seek), reload the page (you must stay logged in), ask a question on `/ask` (the agent trace should
stream step by step).

Check that the video really comes from private S3 and supports Range requests: copy the stream URL from DevTools > Network
(`/api/videos/<id>/stream`) and run

```bash
curl -s -o /dev/null -D - -r 0-99 "<presigned-url>"       # HTTP/1.1 206 Partial Content, Content-Range: bytes 0-99/...
curl -s -o /dev/null -w '%{http_code}\n' "https://YOUR-BUCKET.s3.ap-south-1.amazonaws.com/raw/anything.mp4"   # 403: not public
```

---

## 4. Update to a new version (server)

```bash
cd ~/svs && ./scripts/deploy.sh
```

That is `git pull --ff-only`, `build`, `up -d`, then the status. Only changed services restart. If a release changes `db/init.sql`,
its notes give the `ALTER` statement; run it by hand (init.sql does not re-run on an existing volume):

```bash
dc exec postgres psql -U svs -d svs -P pager=off -c "ALTER TABLE ...;"
```

Changing `WHISPER_SIZE`, `EMBED_MODEL` or `MAX_UPLOAD_MB` needs the rebuild too (models and the UI limit are baked into the images).
A different embedding dimension also needs a schema change and a re-ingest.

---

## 5. Backup and restore

```bash
cd ~/svs
./scripts/backup.sh            # backups/svs-<UTC timestamp>.dump, newest 7 kept
./scripts/backup.sh --s3       # also copies it to s3://YOUR-BUCKET/backups/  (needs: sudo snap install aws-cli --classic)
```

Nightly: `crontab -e`, add `15 2 * * * cd /home/ubuntu/svs && ./scripts/backup.sh --s3 >> backups/cron.log 2>&1`.
The role policy above already allows `PutObject` on the whole bucket; add an S3 lifecycle rule on the `backups/` prefix to expire old
copies. Videos are not in the dump (they are in S3; enable bucket versioning if you want protection against deletes).

**Restore** (replaces the current database contents):

```bash
cd ~/svs
# optional: aws s3 cp s3://YOUR-BUCKET/backups/svs-20260101-021500.dump backups/
dc stop api worker beat
dc exec -T postgres psql -U svs -d postgres -P pager=off -c "DROP DATABASE svs WITH (FORCE);" -c "CREATE DATABASE svs OWNER svs;"
dc exec -T postgres psql -U svs -d svs -c "CREATE EXTENSION IF NOT EXISTS vector;"
dc exec -T postgres pg_restore -U svs -d svs --no-owner < backups/svs-20260101-021500.dump
dc start api worker beat
```

Test a restore once on a scratch machine before you rely on it.

---

## 6. Stop the instance to save cost

Stopping removes the compute charge. You still pay for the EBS volume, the S3 data, and an Elastic IP that is **not** attached to a
running instance (a small hourly fee).

```bash
cd ~/svs && ./scripts/backup.sh --s3     # optional safety copy
dc stop                                  # clean container shutdown
```

Then stop the instance (laptop): `aws ec2 stop-instances --region ap-south-1 --instance-ids i-0123456789abcdef0`.
Start it again with `aws ec2 start-instances ...`; everything comes back by itself (`restart: unless-stopped`, Docker starts on boot).
Keep the Elastic IP: the hostname, and the certificate stored in the `caddy_data` volume, stay valid. Without it the IP changes,
so does the sslip.io name, and you must update `DOMAIN` and get a new certificate.

---

## Troubleshooting

Start with `dc ps` and `dc logs --tail 100 <service>` (`caddy`, `api`, `worker`, `beat`, `postgres`, `redis`).

**Certificate not issued** (browser says "not secure" or the connection resets)
* `dc logs caddy | grep -iE 'obtain|error|acme'`.
* Ports 80 *and* 443 must be open to the world in the security group (the HTTP-01 challenge uses 80), and `DOMAIN` must be the dashed
  **Elastic IP** (`nslookup <DOMAIN>` returns your IP).
* sslip.io is a shared domain, so Let's Encrypt may rate-limit it. Caddy then falls back to ZeroSSL by itself; wait a few minutes and
  read the log. Do not delete the `caddy_data` volume while experimenting: that discards the certificates and burns more rate limit.
* After fixing the cause: `dc restart caddy`.

**502 from caddy**
* The api is down or still booting (each gunicorn worker imports torch: 30-60 s). `dc ps`, then `dc logs --tail 100 api`.
* api exits with `Refusing to start with APP_ENV=production: ...`: fix that value in `.env.prod`, then `dc up -d`.
* api exits with a `StorageError`: see the S3 item below.
* `password authentication failed`: `POSTGRES_PASSWORD` and the password in `DATABASE_URL` differ, or the password was changed after the
  volume was created (the database keeps the old one). Fix the file, or `dc exec postgres psql -U svs -c "ALTER USER svs PASSWORD '...'"`.
* `OOMKilled`: `docker inspect --format '{{.State.OOMKilled}}' $(dc ps -q api)`; see the next item.

**Worker out of memory** (job stuck at *transcribing*, worker restarting, `dmesg | grep -i oom`)
* The worker has a 2 GB cap so it cannot take the database down with it. Check `free -h` (swap on) and `docker stats --no-stream`.
* Keep `WHISPER_SIZE=base` (or `tiny`) on 4 GB, keep the worker at `--concurrency=1`, upload one video at a time.
* Set `WEB_CONCURRENCY=1` in `.env.prod` to give the worker more room, or move to a t3.large.
* Re-run failed jobs with *Reprocess* on `/admin`.

**S3 `AccessDenied` or "Unable to locate credentials"**
* "Unable to locate credentials": the container cannot reach the instance role. Check that the instance has the `svs-ec2` profile
  (`aws ec2 describe-iam-instance-profile-associations --region ap-south-1`) and that the IMDS hop limit is 2 (step 1.3). Test:
  `dc exec api python -c "import boto3; print(boto3.Session().get_credentials())"` must not print `None`.
  Also make sure `.env.prod` has no stray `S3_ACCESS_KEY` / `S3_SECRET_KEY` (compose blanks them, but check anyway).
* `AccessDenied` at startup or on upload: the policy needs `s3:ListBucket` on `arn:aws:s3:::BUCKET` and the object actions on
  `arn:aws:s3:::BUCKET/*` (two different resources). Check the bucket name and `AWS_REGION` (must be the bucket's region).
* `NoSuchBucket` / "does not exist": the bucket is not in `AWS_REGION`; production never creates it.

**Video won't play**
* DevTools > Network: `/api/videos/<id>/stream` must return 200 with an `https://<bucket>.s3.ap-south-1.amazonaws.com/...` URL.
* The media request itself: `403` with `Request has expired` / `SignatureDoesNotMatch`: the instance clock is wrong (`timedatectl`) or the
  role's temporary credentials expired (the player refetches the URL once by itself). `301/307`: wrong `AWS_REGION`. `404`: the object is
  missing (failed upload, wrong bucket).
* Plays but cannot seek: the response must be `206` with `Accept-Ranges: bytes` (the Range test in section 3).
  Some `.mkv`/`.mov` files are not playable by browsers; MP4 (H.264/AAC) and WebM are.
* An upload ends in `413`: raise `MAX_UPLOAD_MB` in `.env.prod`, then `./scripts/deploy.sh --no-pull`.

**Logged out on refresh**
* The refresh cookie is `Secure`, `HttpOnly`, `SameSite=Lax`, path `/auth`. It only works when the page is opened as
  `https://<DOMAIN>`: not `http://`, not the bare IP, not another hostname.
* DevTools > Application > Cookies must show `refresh_token` after login; Network: `POST /auth/refresh` on reload must be 200.
  A `401` with the cookie present means a used token was replayed, which revokes all sessions (two browsers sharing the cookie, a
  retried request). Log in again.
* `COOKIE_SECURE=false` is refused at startup in production; `true` over plain HTTP makes the browser drop the cookie.
