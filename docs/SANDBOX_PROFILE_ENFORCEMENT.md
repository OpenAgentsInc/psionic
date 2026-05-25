# Sandbox Profile Enforcement

> Status: implemented for `psionic-sandbox` profile-bound local execution and
> refusal receipts.

`psionic-sandbox` executes bounded jobs only through a declared
`ProviderSandboxProfile`. The profile is the authority for execution class,
network policy, filesystem policy, timeout limit, artifact policy, and secret
policy.

## Enforcement

`execute_sandbox_job` rejects a request before runtime start when:

- the requested execution class does not match the profile;
- the requested timeout exceeds the profile timeout limit;
- requested CPU, memory, or disk limits exceed the profile;
- requested network policy differs from the declared profile policy;
- requested filesystem policy differs from the declared profile policy;
- the profile forbids injected environment or secret material;
- expected output paths escape the workspace.

Policy refusals return `ProviderSandboxExecutionReceipt` with
`final_state = rejected` and `termination_reason = policy_rejected`.

## Receipt Evidence

Every execution receipt now carries both the declared profile boundary and the
request boundary:

- `profile_id`
- `profile_digest`
- `declared_network_policy`
- `requested_network_policy`
- `declared_filesystem_policy`
- `requested_filesystem_policy`
- `declared_timeout_limit_s`
- `requested_timeout_s`
- `artifact_output_policy`
- `secret_policy`

This lets schedulers and settlement layers verify that a job ran under the
declared sandbox profile or was refused before it could escape that profile.

## Boundary

Psionic owns runtime profile enforcement and execution evidence. Higher-level
assignment admission, workroom policy, user acceptance, and settlement remain
outside this repo.
