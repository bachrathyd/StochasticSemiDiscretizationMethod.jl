using Documenter
using StochasticSemiDiscretizationMethod

DocMeta.setdocmeta!(StochasticSemiDiscretizationMethod, :DocTestSetup,
                    :(using StochasticSemiDiscretizationMethod); recursive = true)

makedocs(;
    modules  = [StochasticSemiDiscretizationMethod],
    authors  = "Henrik T. Sykora and Dániel Bachrathy",
    sitename = "StochasticSemiDiscretizationMethod.jl",
    format   = Documenter.HTML(;
        canonical = "https://bachrathyd.github.io/StochasticSemiDiscretizationMethod.jl",
        prettyurls = get(ENV, "CI", "false") == "true",
        edit_link = "master",
        assets = String[],
    ),
    pages = [
        "Home"          => "index.md",
        "Examples"      => "examples.md",
        "API reference" => "api.md",
    ],
    # Re-exported symbols (ProportionalMX, DelayMX, …) are documented in the
    # SemiDiscretizationMethod package, so keep missing-docs / cross-references
    # non-fatal for the build.
    warnonly = [:missing_docs, :cross_references],
)

deploydocs(;
    repo      = "github.com/bachrathyd/StochasticSemiDiscretizationMethod.jl",
    devbranch = "master",
    push_preview = true,
)
