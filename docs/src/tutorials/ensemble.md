# Ensemble modeling

In this tutorial, we show how EasyModelAnalysis handles collections of models for comparitive analysis and joint-fitting.

We will start with a collection of simple petri nets from the AlgebraicPetri.jl package, fit each to a dataset, and then compare the models.

Then we move on to MIRA nets, which allow for more complex rate laws in transitions. 

We 

First we import the necesary libraries
<!-- ```julia
using EasyModelAnalysis
using DataFrames, AlgebraicPetri, Catlab
using Catlab.CategoricalAlgebra: read_json_acset
using Setfield
using MathML, JSON3
using CommonSolve -->

Then we define some helper functions for loading the data 

```julia
# rescale data to be proportion of population
function scale_df!(df)
    for c in names(df)[2:end]
        df[!, c] = df[!, c] ./ total_pop
    end
end
function CommonSolve.solve(sys::ODESystem; prob_kws = (;), solve_kws = (;), kws...)
    solve(ODEProblem(sys; prob_kws..., kws...); solve_kws..., kws...)
end
getsys(sol) = sol.prob.f.sys
getsys(prob::ODEProblem) = prob.f.sys
read_replace_write(fn, rs) = write(fn, replace(read(fn, String), rs...))

```