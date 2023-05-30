# im putting them in EMA for now to revise, but will move to docs when done. remeber to rm these deps
using Downloads, CSV, URIs, DataFrames, Dates
using AlgebraicPetri
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

"""
Transform list of args into Symbolics variables     
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

sys_syms(sys) = [states(sys); parameters(sys)]
remove_t(x) = Symbol(replace(String(x), "(t)" => ""))

function set_sys_defaults(sys, pn)
    pn_defs = get_defaults(pn)
    syms = sys_syms(sys)
    defs = _symbolize_args(pn_defs, syms)
    sys = ODESystem(pn; defaults = defs)
end

# function ModelingToolkit.ODESystem(pn::PropertyLabelledReactionNet)
#     sys = ODESystem(pn)
#     sys = set_sys_defaults(sys, pn)
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

# "helper to make global_datafit "
# function make_bounds(sys; st_bound = (0.0, 1), p_bound = (0.0, 1.0))
#     st_space = states(sys) .=> (st_bound,)
#     p_space = parameters(sys) .=> (p_bound,)
#     [st_space; p_space]
# end

function plot_covidhub(df; labs = ["incident deaths", "incident hosp", "incident cases"],
                       kws...)
    plt = plot(kws...)
    plot!(plt, df.t, df.deaths; label = labs[1], color = "blue")
    plot!(plt, df.t, df.hosp; label = labs[2], color = "orange")
    plot!(plt, df.t, df.cases; label = labs[3], color = "green")
    plt
end