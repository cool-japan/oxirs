# OxiRS RFC Process

The RFC (Request for Comments) process is used for proposing and discussing major changes to the OxiRS ecosystem. This includes new features, architectural changes, and significant modifications to existing functionality.

## RFC Process

1. **Draft**: Create a new RFC document in the `drafts/` directory using the template
2. **Discussion**: Open a GitHub issue or discussion for community feedback
3. **Review Period**: 14-day comment window with lazy consensus
4. **Decision**: The RFC is either accepted, rejected, or requires more work
5. **Implementation**: Accepted RFCs move to `accepted/` and can be implemented

## Directory Structure

- `drafts/`: New RFC proposals under active discussion
- `accepted/`: Approved RFCs that guide implementation
- `rejected/`: RFCs that were not accepted (preserved for historical reference)
- `template.md`: Standard template for new RFCs

## RFC Numbering

RFCs are numbered sequentially:
- `0001-core-architecture.md`
- `0002-graphql-mapping.md`
- `0003-vector-search.md`

## Creating a New RFC

1. Copy `template.md` to `drafts/NNNN-title.md`
2. Fill out all sections of the template
3. Submit a pull request
4. Open a discussion issue
5. Gather feedback and iterate

## Guidelines

- Be specific and detailed in your proposals
- Consider backwards compatibility
- Include examples and use cases
- Address potential drawbacks and alternatives
- Keep the scope focused and manageable

## Current Active RFCs

Check the `drafts/` directory for RFCs currently under discussion.