@testset "Citation metadata" begin
    repo_root = normpath(joinpath(@__DIR__, ".."))
    read_text(path) = replace(read(path, String), "\r\n" => "\n")

    project = read_text(joinpath(repo_root, "Project.toml"))
    citation = read_text(joinpath(repo_root, "CITATION.cff"))
    readme = read_text(joinpath(repo_root, "README.md"))
    checklist = read_text(joinpath(repo_root, ".github", "RELEASE_CHECKLIST.md"))

    concept_doi = "10.5281/zenodo.7535883"
    upstream_paper = "https://arxiv.org/abs/2405.08810"
    ecosystem_doi = "10.1080/10556788.2026.2702926"
    version = match(r"(?m)^version\s*=\s*\"([^\"]+)\"", project).captures[1]

    @test occursin("doi: \"$concept_doi\"", citation)
    @test !occursin("identifiers:", citation)
    @test occursin("version: \"$version\"", citation)
    @test occursin(upstream_paper, citation)
    @test occursin(ecosystem_doi, citation)

    @test occursin("badge/DOI/$concept_doi.svg", readme)
    @test occursin("doi.org/$concept_doi", readme)
    @test occursin(upstream_paper, readme)
    @test occursin(ecosystem_doi, readme)
    @test occursin(
        "does not identify or\narchive upstream Qiskit components",
        readme,
    )

    @test occursin(concept_doi, checklist)
    @test occursin("Can manage", checklist)
    @test occursin("JuliaQUBO/QUBO.jl#66", checklist)
    @test occursin("cffconvert --validate --infile CITATION.cff", checklist)
end
