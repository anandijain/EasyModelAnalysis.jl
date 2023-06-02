# im putting them in EMA for now to revise, but will move to docs when done. remeber to rm these deps
using Downloads, CSV, URIs, DataFrames, Dates
using AlgebraicPetri
import Catlab.ACSetInterface: has_subpart
using MathML
using Setfield
using JSON3
function download_data(url, dd)
    filename = joinpath(dd, URIs.unescapeuri(split(url, "/")[end]))
    if !isfile(filename)
        Downloads.download(url, filename)
    end
    filename
end

function get_covidhub_data(url, dd)
    return CSV.read(download_data(url, dd), DataFrame)
end

select_location(df, location) = df[df.location .== location, :]
select_location(location) = df -> select_location(df, location)

function date_join(colnames, dfs...)
    d_ = innerjoin(dfs..., on = :date, makeunique = true)
    d = d_[:, colnames]
    return sort!(d, :date)
end

function groupby_week(df)
    first_monday = first(df.date) - Day(dayofweek(first(df.date)) - 2) % 7
    df.t = (Dates.value.(df.date .- first_monday) .+ 1) .÷ 7

    weekly_summary = combine(groupby(df, :t),
                             :cases => sum,
                             :deaths => sum,
                             :hosp => sum)

    rename!(weekly_summary, [:t, :cases, :deaths, :hosp])
    weekly_summary
end

"""
Separate keys and values    
"""
_unzip(d::Dict) = (collect(keys(d)), collect(values(d)))
"""
Unzip a collection of pairs    
"""
unzip(ps) = first.(ps), last.(ps)

remove_t(x) = Symbol(replace(String(x), "(t)" => ""))

function set_sys_defaults(sys, pn; kws...)
    pn_defs = get_defaults(pn)
    syms = sys_syms(sys)
    defs = _symbolize_args(pn_defs, syms)
    sys = ODESystem(pn; tspan = ModelingToolkit.get_tspan(sys), defaults = defs, kws...)
end

"""
Transform list of args into Symbolics variables 
```julia
@parameters sig

_symbolize_args([:sig => 1], [sig])

Dict{Num, Int64} with 1 entry:
  sig => 1
```
"""
function _symbolize_args(incoming_values, sys_vars)
    pairs = collect(incoming_values)
    ks, values = unzip(pairs)
    symbols = Symbol.(ks)
    vars_as_symbols = Symbolics.getname.(sys_vars)
    symbols_to_vars = Dict(vars_as_symbols .=> sys_vars)
    Dict([symbols_to_vars[vars_as_symbols[findfirst(x -> x == symbol, vars_as_symbols)]]
          for symbol in symbols] .=> values)
end

function get_defaults(pn)
    [snames(pn) .=> collect(pn[:concentration]); tnames(pn) .=> collect(pn[:rate])]
end

# this 
sys_syms(sys) = [states(sys); parameters(sys)]

function get_rn_defaults(sys, rn)
    _symbolize_args(get_defaults(rn), sys_syms(sys))
end

function generate_sys_args(p::AbstractPetriNet)
    t = first(@variables t)
    sname′(i) =
        if has_subpart(p, :sname)
            sname(p, i)
        else
            Symbol("S", i)
        end
    tname′(i) =
        if has_subpart(p, :tname)
            tname(p, i)
        else
            Symbol("r", i)
        end

    S = [first(@variables $Si(t)) for Si in sname′.(1:ns(p))]
    r = [first(@parameters $ri) for ri in tname′.(1:nt(p))]
    D = Differential(t)

    tm = TransitionMatrices(p)

    coefficients = tm.output - tm.input

    transition_rates = [r[tr] * prod(S[s]^tm.input[tr, s] for s in 1:ns(p))
                        for tr in 1:nt(p)]

    eqs = [D(S[s]) ~ transition_rates' * coefficients[:, s] for s in 1:ns(p)]

    eqs, t, S, r
end

function ModelingToolkit.ODESystem(rn::AbstractLabelledReactionNet; name = :ReactionNet,
                                   kws...)
    sys = ODESystem(generate_sys_args(rn)...; name = name, kws...)
    defaults = get_rn_defaults(sys, rn)
    @set! sys.defaults = defaults
    sys
end

# function ModelingToolkit.ODESystem(rn::PropertyLabelledReactionNet; name = :ReactionNet, kws...)
#     sys = ODESystem(generate_sys_args(rn)...; name = name, kws...)
#     defaults = get_rn_defaults(sys, rn)
#     @set! sys.defaults = defaults
#     sys
# end

function st_defs(sys)
    filter(x -> !ModelingToolkit.isparameter(x[1]),
           collect(ModelingToolkit.defaults(sys)))
end
function p_defs(sys)
    filter(x -> ModelingToolkit.isparameter(x[1]),
           collect(ModelingToolkit.defaults(sys)))
end

to_data(df, mapping) = [k => df[:, v] for (k, v) in mapping]

"specify how many weeks out of the df to use for calibration/datafitting"
function train_test_split(dfi; train_weeks = nrow(dfi) ÷ 2)
    @assert train_weeks < nrow(dfi)
    dfi[1:train_weeks, :], dfi[(train_weeks + 1):end, :]
end

function select_timeperiods(df::DataFrame, split_length::Int; step::Int = split_length)
    if split_length < 1
        error("Split length must be a positive integer.")
    end
    if step < 1
        error("Step must be a positive integer.")
    end
    return [df[i:(i + split_length - 1), :]
            for i in 1:step:(nrow(df) - split_length + 1)]
end

function plot_covidhub(df; labs = ["incident deaths", "incident hosp", "incident cases"],
                       kws...)
    plt = plot(kws...)
    plot!(plt, df.t, df.deaths; label = labs[1], color = "blue")
    plot!(plt, df.t, df.hosp; label = labs[2], color = "orange")
    plot!(plt, df.t, df.cases; label = labs[3], color = "green")
    plt
end

function replace_nothings_with_defaults!(sys::ModelingToolkit.ODESystem)
    defaults_dict = ModelingToolkit.defaults(sys)
    for (key, value) in defaults_dict
        isnothing(value) && (defaults_dict[key] = 0.01)
    end
    sys
end

function ModelingToolkit.ODESystem(p::PropertyLabelledReactionNet;
                                   name = :PropMiraNet, kws...)
    t = first(@variables t)

    sname′(i) =
        if has_subpart(p, :sname)
            sname(p, i)
        else
            Symbol("S", i)
        end
    tname′(i) =
        if has_subpart(p, :tname)
            tname(p, i)
        else
            Symbol("r", i)
        end

    S = [first(@variables $Si(t)) for Si in sname′.(1:ns(p))]
    S_ = [first(@variables $Si) for Si in sname′.(1:ns(p))] # MathML doesn't know whether a Num should be dependent on t, so we use this to substitute 
    st_sub_map = S_ .=> S

    # we have rate parameters and then the mira_parameters
    r = [first(@parameters $ri) for ri in tname′.(1:nt(p))]

    js = [JSON3.read(tprop(p, ti)["mira_parameters"]) for ti in 1:nt(p)]
    for si in 1:ns(p)
        x = get(sprop(p, si), "mira_parameters", nothing)
        isnothing(x) && continue
        push!(js, JSON3.read(x))
    end

    mira_ps = merge(js...)

    mira_st_ps = [first(@variables $k = v) for (k, v) in mira_ps]
    mira_p = ModelingToolkit.toparam.(mira_st_ps)

    ps_sub_map = mira_st_ps .=> mira_p
    # mira_p = [first(@parameters $k = v) for (k, v) in mira_ps]

    D = Differential(t)

    tm = TransitionMatrices(p)

    coefficients = tm.output - tm.input
    sym_rate_exprs = [substitute(MathML.parse_str(tprop(p, tr)["mira_rate_law_mathml"]),
                                 st_sub_map) for tr in 1:nt(p)]

    mrl_vars = union(Symbolics.get_variables.(sym_rate_exprs)...)
    sts_that_should_be_ps = setdiff(mrl_vars, S)
    # sts_that_should_be_ps2 = setdiff(sts_that_should_be_ps, mira_st_ps)
    # append!(mira_p, ModelingToolkit.toparam.(sts_that_should_be_ps2)) 
    # append!(ps_sub_map, sts_that_should_be_ps2 .=> ModelingToolkit.toparam.(sts_that_should_be_ps2)) # why ben, why!! XXlambdaXX

    # @info "this is weird, mira parameters are named differently than the tname"
    # mp_js = map(x->JSON3.read(x["mira_parameters"]), tps)
    # mps = merge(Dict.(mp_js)...)
    # @assert  allequal(length.(mps)) && length(mps[1]) == 1
    # r_ = [first(@parameters $p = def) for (p, def) in mps]

    
    # i dont know if i need to be explicit here 
    @show mira_p  mira_ps
    default_p = mira_p .=> ModelingToolkit.getdefault.(mira_p)
    default_u0 = S .=> p[:concentration]
    defaults = [default_p; default_u0]
    # to_ps_names = Symbolics.getname.(sts_that_should_be_ps)
    # ps_sub_map = sts_that_should_be_ps .=> ModelingToolkit.toparam.(sts_that_should_be_ps)

    full_sub_map = [st_sub_map; ps_sub_map]
    sym_rate_exprs = [substitute(sym_rate_expr, ps_sub_map)
                      for sym_rate_expr in sym_rate_exprs]

    tm = TransitionMatrices(p)

    coefficients = tm.output - tm.input

    # og 
    # transition_rates = [r[tr] * prod(S[s]^tm.input[tr, s] for s in 1:ns(p))
    #                     for tr in 1:nt(p)]

    # transition_rates= [sym_rate_exprs[tr] * prod(S[s]^tm.input[tr, s] for s in 1:ns(p))
    #                     for tr in 1:nt(p)]

    # disabling this for now
    transition_rates = [r[tr]* sym_rate_exprs[tr] for tr in 1:nt(p)]
    observable_species_idxs = filter(i -> sprop(p, i)["is_observable"], 1:ns(p))
    observable_species_names = Symbolics.getname.(S[observable_species_idxs])

    # r_ = [first(@variables $ri(t)) for ri in tname′.(1:nt(p))]

    # flux_eqs = [t_state ~ t_rate for (t_state, t_rate) in zip(r_, transition_rates)]

    # todo, this really needs to be validated, since idk if it's correct, update: pretty sure its busted.
    deqs = [D(S[s]) ~ transition_rates' * coefficients[:, s]
            for s in 1:ns(p) if Symbolics.getname(S[s]) ∉ observable_species_names]

    obs_eqs = [substitute(S[i] ~ Symbolics.parse_expr_to_symbolic(Meta.parse(sprop(p, i)["expression"]),
                                                                  @__MODULE__),
                          Dict(full_sub_map))
               for i in observable_species_idxs]

    # eqs = Equation[flux_eqs; deqs; obs_eqs]
    eqs = Equation[deqs; obs_eqs]
    # eqs = map(identity, eqs)
    # sys = ODESystem(eqs, t; name = name)
    sys = ODESystem(eqs, t; name = name, defaults = Dict(defaults),
                    kws...)
end

function ModelingToolkit.ODESystem(p::PropertyLabelledReactionNet; name = :MiraNet, kws...)
    t = first(@variables t)
    D = Differential(t)
    tm = TransitionMatrices(p)
    coefficients = tm.output - tm.input

    sname′(i) =
        if has_subpart(p, :sname)
            sname(p, i)
        else
            Symbol("S", i)
        end
    tname′(i) =
        if has_subpart(p, :tname)
            tname(p, i)
        else
            Symbol("r", i)
        end

    S = [first(@variables $Si(t)) for Si in sname′.(1:ns(p))]
    S_ = [first(@variables $Si) for Si in sname′.(1:ns(p))] # MathML doesn't know whether a Num should be dependent on t, so we use this to substitute 
    st_sub_map = S_ .=> S

    t_ps, flux_eqs = parse_tprops(p)
    s_ps = union(parse_prop_parameters.(sprops(p))...)
    ps = union(t_ps, s_ps)
    ps_sub_vars = [only(@variables $x) for x in Symbolics.getname.(ps)]
    ps_sub_map = ps_sub_vars .=> ps

    subs = [ps_sub_map; st_sub_map]
    subd = Dict(subs)

    flux_eqs = map(x -> substitute(x, subd), flux_eqs)
    tvars = ModelingToolkit.lhss(flux_eqs)

    default_p = ps .=> ModelingToolkit.getdefault.(ps)
    default_u0 = S .=> p[:concentration]
    defaults = Dict([default_p; default_u0])

    observable_species_idxs = filter(i -> sprop(p, i)["is_observable"], 1:ns(p))
    observable_species_names = Symbolics.getname.(S[observable_species_idxs])

    # i don't understand where p[:rate] comes into play. it seems like rate is only needed if there aren't custom rate laws
    deqs = [D(S[s]) ~ tvars' * coefficients[:, s]
            for s in 1:ns(p) if Symbolics.getname(S[s]) ∉ observable_species_names]

    # there should be a mathml but there isnt
    obs_eqs = [substitute(S[i] ~ Symbolics.parse_expr_to_symbolic(Meta.parse(sprop(p, i)["expression"]),
                                                                  @__MODULE__),
                          Dict(subs))
               for i in observable_species_idxs]

    eqs = Equation[flux_eqs; deqs; obs_eqs]
    ODESystem(eqs, t; name, defaults, kws...)
end

"can give it a S or T and itll make the pars from the prop dict"
function parse_prop_parameters(prop)
    pars = []
    !haskey(prop, "mira_parameters") && (return pars)

    mps = JSON3.read(prop["mira_parameters"])
    dists = JSON3.read(prop["mira_parameter_distributions"])
    for (k, v) in mps
        d = dists[k]
        pname = Symbol(k)
        if !isnothing(d) && haskey(d, "parameters")
            d_ps = d["parameters"]
            @assert d["type"] == "StandardUniform1"
            b = (d_ps["minimum"], d_ps["maximum"])

            par = only(@parameters $pname=v [bounds = b])
        else
            par = only(@parameters $pname = v)
        end
        push!(pars, par)
    end
    pars
end

function parse_tprops(p)
    mira_ps = Set()
    tps = tprops(p)
    tp = first(tps)
    flux_eqs = Equation[]
    for i in 1:nt(p)
        eq, pars = parse_tprop(p, i)
        push!(flux_eqs, eq)

        [push!(mira_ps, x) for x in pars] # no append!(set, xs)??
    end
    collect(mira_ps), flux_eqs
end

# doesn't really take a tprop, but the index
function parse_tprop(p, i)
    tp = tprop(p, i)
    tn = tname(p, i)

    pars = parse_prop_parameters(tp)

    t = only(@parameters t) # idc 
    tvar = only(@variables $tn(t))

    rl = MathML.parse_str(tp["mira_rate_law_mathml"])
    # push!(flux_eqs, )
    eq = tvar ~ rl
    eq, pars
end

# filter(ModelingToolkit.isparameter, union(ModelingToolkit.get_variables.(ModelingToolkit.equations(sys))...))

function mira_ps(mn)
    mp_js = map(x -> JSON3.read(x["mira_parameters"]), tprops(mn))
    # mps = merge(Dict.(mp_js)...)
end
""
function remake_for_df(prob, df; u0_fn = nothing)
    remake_workaround!(prob, df, col_st_map)
    remake(prob; tspan = extrema(df.t))
end

"""

user must write this function, we aren't guaranteed that the firsts of the mapping are states, they can be exprs, so we cant use the first row of the df to assign u0s
this is incorrect for the purposes of this exercise, we say that the the first row of the df goes assigned by col_st_map
    

you need one for every model in the ensemble im realizing. ideally the data and model aren't inconsistent 
    """
function remake_workaround!(prob, df, col_st_map)
    cols, u0_vars = EMA.unzip(col_st_map)
    idxs = [findfirst(isequal(x), states(sys)) for x in u0_vars]
    prob.u0[idxs] = collect(df[1, last.(cols)])
    nothing
end

function myl2loss(pvals, (prob, pkeys, t, data))
    p = Pair.(pkeys, pvals)
    prob = remake(prob, tspan = extrema(t), p = p)
    sol = solve(prob, saveat = t)
    tot_loss = 0.0
    for pairs in data
        tot_loss += sum((sol[pairs.first] .- pairs.second) .^ 2)
    end
    return tot_loss
end

"there seems to be a problem when you try to optimize on the observed states  could be due to the u0 updating.
todo the prob's sys should already have the bounds and the tunables set. 
"
function global_ensemble_fit(prob_mapping_ps, dfs, bounds; kws...)
    # @assert allequal(Set.(last.(prob_mapping_ps)))
    # cols, sts = unzip(col_st_map)
    all_gress = []
    for ((prob, mapping), bound) in zip(prob_mapping_ps, bounds)
        sys = prob.f.sys
        sts, cols = unzip(mapping)
        gress = []
        for df in dfs
            # prob = remake_for_df(prob, df)
            # @info prob.u0
            r = df[1, cols]
            new_u0 = sts .=> collect(r)
            # @info new_u0
            prob = remake(prob; u0 = new_u0)
            # @info prob.u0
            fit = global_datafit(prob, bound, df.t, to_data(df, mapping); kws...)
            # @info "THE FIT" fit
            push!(gress, fit)
        end
        push!(all_gress, gress)
    end
    return all_gress
end

# function ensemble_global_datafit()