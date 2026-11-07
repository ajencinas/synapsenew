# Security policy

This repository contains source code only. Do not commit credentials, private
corpora, model checkpoints, or generated evaluation data.

- Report vulnerabilities privately through GitHub Security Advisories:
  <https://github.com/ajencinas/synapsenew/security/advisories/new>
- Do not include secrets, personal paths, or private dataset contents in a
  public issue.
- The checked-in `.env.example` contains placeholders only. Real values belong
  in a gitignored `.env` or your environment/CI secret store.

The public pipeline reads provider keys from environment variables. No API key
value is required to inspect the code or run the offline unit tests.
