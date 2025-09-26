using Pkg

function install_pkgs(pkgs...)
  Pkg.update()
  installedpkgs = [x.name for x in values(Pkg.dependencies())]
  missingpkgs = setdiff(pkgs, installedpkgs)
  if length(missingpkgs)>0
    println("Installing $(missingpkgs)")
    Pkg.add(missingpkgs)
  end
end
## Some of the packages i commonly use...
## install_pkgs("DataFrames","Distributions","GLM","MixedModels","Plots","StatsBase","RDatasets", "JuMP", "XGBoost")

