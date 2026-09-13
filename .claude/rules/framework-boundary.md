# Framework and example ownership

Apply this rule to the entire PR diff against main, including earlier commits.

## Framework admission

`kairyu/` owns independently reusable inference, transport, orchestration,
resource-accounting, and lifecycle contracts. Before changing it, identify:

1. The shared contract that is missing or broken in main, with its code path.
2. Why an existing extension point cannot satisfy that contract.
3. A concrete use and observable regression independent of the target example.
4. The smallest shared mechanism needed, and the policy that stays in the example.

Generic names, configurable switches, examples with arbitrary role names, and
speculative future reuse are not evidence of genericity. Do not disguise an
example workflow as a framework feature. A non-generalizable framework change
is prohibited; implement the example-owned behavior through the appropriate
extension instead. Explain and obtain explicit authorization before expanding
framework scope beyond the shared contracts already authorized in the session.

## Example ownership

Examples own model/GPU choices, runtime-specific adaptations, role prompts,
reasoning budgets, workflow order, review and audit protocols, domain schemas,
document traversal and editing strategies, retries, and publication policy.
Their exclusive helpers, tests, dependencies, and image build settings belong
with them. Storage or execution primitives must not implicitly select these
policies. A generic quota mechanism may know reservation, dispatch, and usage;
it must not know which candidate, review, audit, or page is being executed.

Prefer existing extension points and narrow programmatic dependency injection.
Construct example orchestration with Kairyu's DSL and configuration; do not add
Python orchestration implementations to examples. Missing shared capabilities
must satisfy the framework admission criteria above.
Do not monkeypatch framework globals, copy an entire framework implementation
into an example, or establish a second input/HTTP/lifecycle stack merely to
avoid a justified shared transport change.

## Review and requirements

Review ownership against the full main-to-PR diff, not only the latest commit
or a model-name search. Prior inclusion in a PR does not exempt a change.
Remove example-specific framework behavior and its exclusive dependencies and
tests together. Follow the test policy in CLAUDE.md; file moves are not new
coverage and static configuration lists do not justify extra tests.

Preserve the user's product requirements while correcting ownership. Do not
solve the boundary problem by skipping an ensemble stage, weakening the user
request, summarizing required source material, or substituting a direct answer.

This file is the single source of this rule. CLAUDE.md imports it; AGENTS.md
only instructs the reader to follow CLAUDE.md. Do not duplicate these rules.
