@time @time_imports using EasyModelAnalysis
using DataFrames, AlgebraicPetri, Catlab
using Catlab.CategoricalAlgebra: read_json_acset
EMA = EasyModelAnalysis
datadir = joinpath(@__DIR__, "../data/")
mkpath(datadir)
# download data from covidhub as dataframes
dfc = EMA.get_covidhub_data("https://github.com/reichlab/covid19-forecast-hub/raw/master/data-truth/truth-Incident%20Cases.csv",
                            datadir)
dfd = EMA.get_covidhub_data("https://github.com/reichlab/covid19-forecast-hub/raw/master/data-truth/truth-Incident%20Deaths.csv",
                            datadir)
dfh = EMA.get_covidhub_data("https://github.com/reichlab/covid19-forecast-hub/raw/master/data-truth/truth-Incident%20Hospitalizations.csv",
                            datadir)

# select location to use
dfc, dfd, dfh = map(EMA.select_location("US"), [dfc, dfd, dfh])

# rename to synchronize dataset column names with models
rename!(dfc, :value => :cases)
rename!(dfd, :value => :deaths)
rename!(dfh, :value => :hosp)

# create combined dataframe joined on date
covidhub = EMA.date_join([:date, :cases, :deaths, :hosp], dfc, dfd, dfh)

# aggregate to week-level data
df = EMA.groupby_week(covidhub)

# collection of models to treat as an ensemble
model_urls = [
    "https://github.com/indralab/mira/raw/main/notebooks/ensemble/BIOMD0000000955_miranet.json",
    "https://github.com/indralab/mira/raw/main/notebooks/ensemble/BIOMD0000000960_miranet.json",
    "https://github.com/indralab/mira/raw/main/notebooks/ensemble/BIOMD0000000983_miranet.json",
]

model_fns = [EMA.download_data(url, datadir) for url in model_urls]

T_PLRN = PropertyLabelledReactionNet{Union{Number, Nothing}, Union{Number, Nothing}, Dict}
petris = read_json_acset.((T_PLRN,), model_fns)
pn = petris[1]
sys = ODESystem(pn)
sys = EMA.set_sys_defaults(sys, pn)
prob = ODEProblem(sys, [], (0, 100))
sol = solve(prob)

syss = structural_simplify.(ODESystem.(petris))
syss = [EMA.set_sys_defaults(sys, pn) for (sys, pn) in zip(syss, petris)]

# starting to fit 
total_pop = 30_000_000
N_weeks = 20;
period_step = 10;
train_weeks = 10; # 10 weeks of training data, 10 weeks of testing

# rescale data to be proportion of population
for c in  names(df)[2:end]
    df[!, c] = df[!, c] ./ total_pop
end


all_ts = df.t
dfs = EMA.select_timeperiods(df, N_weeks; step = period_step)

split_dfs = [EMA.train_test_split(df; train_weeks = train_weeks) for df in dfs]
train_dfs, test_dfs = EMA.unzip(split_dfs)

dfi = dfs[1]
prob = remake(prob; tspan=extrema(dfi.t))
global_data()


@unpack Extinct, Recognized, Infected = sys
# obs_sts = [Deaths, Hospitalizations, Cases]

mapping_ps = [Extinct => :deaths, Infected => :cases, Recognized => :hosp]
mapping = Dict(mapping_)
plot_vars = collect(keys(mapping))

losses = []
logged_p = []

callback = function (p, l)

    push!(losses, deepcopy(l))
    push!(logged_p, deepcopy(p))

    return false
end

solve_kws = (;callback)
pbounds = parameters(sys) .=> ((0., 1.),)
data = EMA.to_data(dfi, mapping)
fit_t = dfi.t

fit = EMA.global_datafit(prob, pbounds, fit_t, data; solve_kws)
prob2 = remake(prob; p = last.(fit))

# instead of fitting u0, now we are going to make an intuitive guess function that assigns u0 from the first row of the train dataframe

#fuckk https://github.com/SciML/ModelingToolkit.jl/issues/2171
new_u0s = plot_vars .=> collect(dfi[1, [:deaths, :cases, :hosp]])

prob2 = remake(prob; u0 = new_u0s)
sol = solve(prob)
plot(sol, idxs=plot_vars)
EMA.plot_covidhub(dfi)


sol2 = solve(prob2)
plt = EMA.plot_covidhub(dfi)
plot!(plt, sol2, idxs=plot_vars)
