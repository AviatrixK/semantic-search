-- Runs once on a fresh Postgres volume (docker-entrypoint-initdb.d, before init.sql).
-- Integration tests use this database; they drop and rebuild its schema, never the dev one.
CREATE DATABASE svs_test;
