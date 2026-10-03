export function checkPassword(password) {
    return [
        { label: 'At least 8 characters', ok: password.length >= 8 },
        { label: 'Contains a letter', ok: /\p{L}/u.test(password) },
        { label: 'Contains a digit', ok: /\p{N}/u.test(password) },
        { label: 'At most 72 bytes', ok: new TextEncoder().encode(password).length <= 72 },
    ];
}
export function passwordIsValid(password) {
    return checkPassword(password).every((c) => c.ok);
}
