#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: scripts/bootstrap-tailnet-key-auth.sh [options] <host>

Options:
  --user <name>     Remote username. Default: $TAILNET_SSH_USERNAME or $USER
  --pubkey <path>   Local public key path. Default: ~/.ssh/id_ed25519.pub

Environment:
  TAILNET_SSH_PASSWORD   Required remote password used only for the bootstrap.
  TAILNET_SSH_USERNAME   Optional default remote username.
EOF
}

remote_user="${TAILNET_SSH_USERNAME:-${USER}}"
pubkey_path="${HOME}/.ssh/id_ed25519.pub"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)
      remote_user="$2"
      shift 2
      ;;
    --pubkey)
      pubkey_path="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      echo "error: unknown option $1" >&2
      usage
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

remote_host="$1"

if [[ -z "${TAILNET_SSH_PASSWORD:-}" ]]; then
  echo "error: TAILNET_SSH_PASSWORD is required" >&2
  exit 1
fi

if [[ ! -f "${pubkey_path}" ]]; then
  echo "error: public key not found at ${pubkey_path}" >&2
  exit 1
fi

askpass_script="$(mktemp "${TMPDIR:-/tmp}/psionic_tailnet_askpass.XXXXXX")"
cleanup() {
  rm -f "${askpass_script}"
}
trap cleanup EXIT

cat >"${askpass_script}" <<'EOF'
#!/bin/sh
printf '%s\n' "${TAILNET_SSH_PASSWORD}"
EOF
chmod 700 "${askpass_script}"

password_ssh() {
  DISPLAY=dummy \
    SSH_ASKPASS="${askpass_script}" \
    SSH_ASKPASS_REQUIRE=force \
    TAILNET_SSH_PASSWORD="${TAILNET_SSH_PASSWORD}" \
    ssh -n \
      -o StrictHostKeyChecking=accept-new \
      -o PubkeyAuthentication=no \
      -o PreferredAuthentications=password,keyboard-interactive \
      -o NumberOfPasswordPrompts=1 \
      "${remote_user}@${remote_host}" "$@"
}

password_ssh_with_stdin() {
  DISPLAY=dummy \
    SSH_ASKPASS="${askpass_script}" \
    SSH_ASKPASS_REQUIRE=force \
    TAILNET_SSH_PASSWORD="${TAILNET_SSH_PASSWORD}" \
    ssh \
      -o StrictHostKeyChecking=accept-new \
      -o PubkeyAuthentication=no \
      -o PreferredAuthentications=password,keyboard-interactive \
      -o NumberOfPasswordPrompts=1 \
      "${remote_user}@${remote_host}" "$@"
}

echo "Bootstrapping key auth for ${remote_user}@${remote_host}"
password_ssh "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"

cat "${pubkey_path}" | password_ssh_with_stdin '
  set -euo pipefail
  tmp_key="$(mktemp "${TMPDIR:-/tmp}/psionic_tailnet_key.XXXXXX")"
  trap "rm -f \"${tmp_key}\"" EXIT
  cat >"${tmp_key}"
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    grep -qxF "${line}" ~/.ssh/authorized_keys || printf "%s\n" "${line}" >> ~/.ssh/authorized_keys
  done <"${tmp_key}"
'

ssh \
  -o BatchMode=yes \
  -o StrictHostKeyChecking=accept-new \
  "${remote_user}@${remote_host}" "hostname" >/dev/null

echo "Key auth verified for ${remote_user}@${remote_host}"
