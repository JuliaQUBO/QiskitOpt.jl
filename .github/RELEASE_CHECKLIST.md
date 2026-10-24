# Release checklist

Use this checklist for every tagged release. The Zenodo concept DOI
`10.5281/zenodo.7535883` is the permanent identifier for the `QiskitOpt.jl`
wrapper; do not create another concept record for a new version. It does not
identify or archive the upstream [Qiskit](https://github.com/Qiskit/qiskit) or
[Qiskit Optimization](https://github.com/qiskit-community/qiskit-optimization)
projects.

## Before tagging

- [ ] Follow the [accepted stewardship decision](https://github.com/JuliaQUBO/QiskitOpt.jl/issues/69#issuecomment-5153545880):
      David Bernal (`@bernalde`) is the sole designated Zenodo owner for now.
      Confirm **Can manage** access to the record and all future versions, or
      record outstanding access recovery/transfer work and the review date in
      the ecosystem stewardship tracker, JuliaQUBO/QUBO.jl#66.
- [ ] Revisit adding a backup manager annually or when maintainership changes.
      A single-owner arrangement is accepted while no other maintainer has
      volunteered; use personal accounts and never share a password or API
      token.
- [ ] Update the version in `Project.toml` and the release date and version in
      `CITATION.cff`.
- [ ] Keep the concept DOI in `CITATION.cff` and the README badge unchanged.
- [ ] Confirm the wrapper creators and their order, affiliations, and ORCIDs.
- [ ] Validate the citation metadata with
      `cffconvert --validate --infile CITATION.cff`.
- [ ] Run the package test suite with
      `julia --project -e 'using Pkg; Pkg.test()'`.

## After publishing the GitHub release

- [ ] Confirm the Zenodo GitHub integration published a **new version** under
      the existing concept record automatically when TagBot created the GitHub
      release. Do not deposit a new upload or concept DOI by hand; if the
      integration is ever disabled, re-enable it rather than archiving
      manually.
- [ ] Confirm the archived Zenodo record matches the official GitHub release's
      tag and version, MIT license, `JuliaQUBO/QiskitOpt.jl` repository URL,
      Julia package UUID (`81b20daf-e62b-4502-a0d1-aa084de80e33`), creators,
      affiliations, and ORCIDs. Leave historical version metadata truthful.
- [ ] Confirm the description and related identifiers attribute upstream Qiskit
      and Qiskit Optimization without implying that the wrapper's DOI archives
      those upstream projects.
- [ ] Record the published Zenodo version DOI in the GitHub release notes.
      Keep the concept DOI in `CITATION.cff`; Zenodo assigns the version DOI
      only after it processes the GitHub release, so do not pin a version DOI
      in the release archive's CFF.
- [ ] Verify the version DOI resolves to that exact archive and the concept DOI
      resolves to the latest archived release:

      ```sh
      curl --fail --location --output /dev/null https://doi.org/<version-doi>
      curl --fail --location --output /dev/null https://doi.org/10.5281/zenodo.7535883
      curl --fail --location https://zenodo.org/api/records/<version-record-id>
      ```

      In the API response, confirm `conceptdoi` is
      `10.5281/zenodo.7535883`, `metadata.version` matches the release tag,
      `metadata.custom["code:codeRepository"]` identifies this repository,
      and the record is the latest version under the concept. If Zenodo responds
      slowly or returns HTTP 429, wait for its `Retry-After` interval and retry
      once before treating the DOI as broken.

- [ ] Download the Zenodo archive, record and verify its checksum, confirm it
      contains the tagged `Project.toml` version, and verify its source commit
      matches the GitHub release tag.
