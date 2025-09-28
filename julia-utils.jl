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

function counts(l::Vector)::Dict{eltype(l),Int64}
  ## like Python's collections.Counter
  ## counts([1,2,2,3,3,3,4,4,4,4]) ==> Dict(1=>1,2=>2,3=>3,4=>4)
  c = Dict(i=>sum(l.==i) for i in unique(l))
  @assert sum(values(c)) == length(l)
  c
end

function prop(l::Vector{<:Real})::Vector{Real}
  # julia> prop([1,2,3,4]) # [0.1,0.2,0.3,0.4]
  p = [i/sum(l) for i in l]
  @assert sum(p)==1.0
  p
end
function prop(d::Dict{<:Any,<:Real})
  ## julia> d = Dict("a"=>2,"b"=>3)
  ## julia> prop(d) # Dict("a"=>0.4, "b"=>0.6)
  s = sum(values(d))
  d = Dict(k=>v/s for (k,v) in d)
  @assert sum(values(d))==1.0
  d
end
function prop(l::Real) Dict(l=>l/l) end ## OK??
