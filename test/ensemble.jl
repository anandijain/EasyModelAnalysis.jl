@time @time_imports using EasyModelAnalysis
using EasyModelAnalysis
using DataFrames, AlgebraicPetri, Catlab
using Catlab.CategoricalAlgebra: read_json_acset
using Setfield

using CommonSolve
function CommonSolve.solve(sys::ODESystem; prob_kws = (;), solve_kws = (;))
    solve(ODEProblem(sys; prob_kws...); solve_kws...)
end
getsys(sol) = sol.prob.f.sys
getsys(prob::ODEProblem) = prob.f.sys
read_replace_write(fn, rs) = write(fn, replace(read(fn, String), rs...))

EMA = EasyModelAnalysis
datadir = joinpath(@__DIR__, "../data/")
mkpath(datadir)

# data prep
total_pop = 300_000_000
N_weeks = 20;
period_step = 10;
train_weeks = 10; # 10 weeks of training data, 10 weeks of testing

# rescale data to be proportion of population
function scale_df!(df)
    for c in names(df)[2:end]
        df[!, c] = df[!, c] ./ total_pop
    end
end

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
orig_df = deepcopy(df) #for future reference
scale_df!(df)

all_ts = df.t
tspan = extrema(df.t)
dfs = EMA.select_timeperiods(df, N_weeks; step = period_step)
split_dfs = [EMA.train_test_split(df; train_weeks = train_weeks) for df in dfs]
train_dfs, test_dfs = EMA.unzip(split_dfs)

dfi = dfs[1]
dftr = train_dfs[1]
dfts = test_dfs[1]

# modeling
# collection of models to treat as an ensemble
# note there is a bug in these models with a faulty parameter name "XXlambdaXX", so we replace them
model_urls = [
    "https://github.com/indralab/mira/raw/main/notebooks/ensemble/BIOMD0000000955_miranet.json",
    "https://github.com/indralab/mira/raw/main/notebooks/ensemble/BIOMD0000000960_miranet.json",
    "https://github.com/indralab/mira/raw/main/notebooks/ensemble/BIOMD0000000983_miranet.json",
]

model_fns = [EMA.download_data(url, datadir) for url in model_urls]
map(x -> read_replace_write(x, ["XXlambdaXX" => "lambda"]), model_fns)

T_PLRN = PropertyLabelledReactionNet{Union{Number, Nothing}, Union{Number, Nothing}, Dict}
pns = read_json_acset.((T_PLRN,), model_fns)
syss = ODESystem.(pns; tspan)

function setup_sys(sys, pn)
    structural_simplify(EMA.replace_nothings_with_defaults!(EMA.set_sys_defaults(sys, pn)))
end

setup_sys(syss[2], pns[2])
syss = [setup_sys(sys, pn) for (sys, pn) in zip(syss, pns)]
syss = complete.(syss)
sys, sys2, sys3 = syss
probs = ODEProblem.(syss)
og_probs = deepcopy.(probs)
sols = solve.(probs; saveat = all_ts)

@unpack Deaths, Hospitalizations, Cases, Extinct, Infected, Threatened = sys
@unpack Deceased, Infectious, Hospitalized = sys2
@unpack Infected_reported, = sys3

obs_sts = Num[Deaths, Hospitalizations, Cases]
display.(plot.(sols; idxs = obs_sts))

mapping = [Deaths => :deaths, Hospitalizations => :hosp, Cases => :cases]

prob_mapping_ps = probs .=> (mapping,)
col_st_map = [:deaths => Extinct, :cases => Infected, :hosp => Threatened]
mapping2 = reverse.(col_st_map)
col_st_map2 = [:deaths => Deceased, :cases => Infectious, :hosp => Hospitalized]
col_st_map3 = [
    :deaths => Deceased,
    :cases => Infected_reported,
    :hosp => Infected_reported * 0.05,
]
cs_maps = [col_st_map, col_st_map2, col_st_map3]
mappings = map(x -> reverse.(x), cs_maps)
prob_mapping_ps = probs .=> mappings

bounds = [parameters(sys) .=> ((0.0, 1.0),) for sys in syss]
# fits = EMA.global_ensemble_fit([prob_mapping_ps[1]], [df], bounds;col_st_map)
# fits = EMA.global_ensemble_fit([prob_mapping_ps[1]], dfs, bounds;col_st_map)

prob = probs[1]
cols, sts = unzip(col_st_map)
r = dfs[1][1, cols]
prob = remake(prob; u0 = sts .=> collect(r), tspan = extrema(dfs[1].t))
plt = EMA.plot_covidhub(dfs[1])
sol = solve(prob; saveat = dfs[1].t)
scatter!(plt, sol; idxs = sts)

fits = EMA.global_ensemble_fit(prob_mapping_ps, [df], bounds; maxiters = 100)

# fits = EMA.global_ensemble_fit(prob_mapping_ps, dfs, bounds; maxiters=100)

ps = ModelingToolkit.parameters(sys)

ModelingToolkit.getbounds(sys, ModelingToolkit.tunable_parameters(sys))
global losses = []
global logged_p = []
global opt_step = 0

callback = function (p, l)
    global opt_step += 1
    if opt_step % 100 == 0
        push!(losses, deepcopy(l))
        push!(logged_p, deepcopy(p))
        # display(plot(losses))
    end
    return false
end

function logged_p_df(pkeys, logged_p)
    DataFrame(stack(logged_p)', Symbolics.getname.(pkeys))
end
# there

# sols = []
function fit_plot(sol, df, sts)
    plt = EMA.plot_covidhub(df)
    plt = scatter!(plt, sol; idxs = sts)
    display(plt)
end
# for lp in logged_p[1:100:end]
#     prob = remake(prob; p = ps .=> lp)
#     sol = solve(prob; saveat = dfs[1].t)
#     fit_plot(sol, dfs[1], sts)
#     push!(sols, sol)
# end

function remake_solve(prob, fit, df, col_st_map)
    cols, sts = EMA.unzip(col_st_map)
    r = df[1, cols]
    new_u0 = sts .=> collect(r)
    @info new_u0
    prob = remake(prob; u0 = new_u0)

    prob = remake(prob; u0 = new_u0, p = fit, tspan = extrema(df.t))
    sol = solve(prob; saveat = df.t)
    fit_plot(sol, df, sts)
    sol
end

fits = EMA.global_ensemble_fit([prob_mapping_ps[1]], dfs[1:5], bounds; col_st_map,
                               solve_kws = (; callback))
plot(losses)
@info prob.p
# prob = remake(prob; p = fits[1][1])
sols = []
for (i, fit) in enumerate(fits[1])
    @info "before plot"
    remake_solve(og_probs[1], parameters(getsys(prob)) .=> og_probs[1].p, dfs[i],
                 col_st_map)
    @info "after fitting plot"
    sol = remake_solve(prob, fits[1][i], dfs[i], col_st_map)
    push!(sols, sol)
end

@info prob.p

sol = solve(prob; saveat = dfs[1].t)

plot(sol, idxs = sts)
solve_kws = (; callback)

ps = first.(bounds)
p_df = logged_p_df(first.(bounds), logged_p)

pbounds = parameters(sys) .=> ((0.0, 1.0),)
data = EMA.to_data(dftr, mapping)
fit_t = dftr.t

prob = remake(prob; tspan = extrema(dftr.t))
fit = EMA.global_datafit(prob, pbounds, fit_t, data; solve_kws)
prob2 = remake(prob; p = last.(fit))
sol = solve(prob)
sol2 = solve(prob2)
plot(sol, idxs = plot_vars)
plot!(sol2, idxs = plot_vars)
# instead of fitting u0, now we are going to make an intuitive guess function that assigns u0 from the first row of the train dataframe
EMA.plot_covidhub(dfi)
new_u0s = plot_vars .=> collect(dfi[1, [:deaths, :cases, :hosp]])

prob3 = deepcopy(prob)
update_u0!(prob3, sys, dfi, mapping_ps)
# prob3 = remake(prob; u0 = new_u0s)
sol = solve(prob3)
plt = EMA.plot_covidhub(dfi)
plot!(sol, idxs = plot_vars)

fit = EMA.global_datafit(prob3, pbounds, fit_t, data; solve_kws)
fit = EMA.global_datafit(prob3, pbounds, fit_t, data; solve_kws)
prob4 = remake(prob3; p = last.(fit))
sol4 = solve(prob4)
plot!(plt, sol4, idxs = plot_vars)

sol2 = solve(prob2)
plt = EMA.plot_covidhub(dfi)
plot!(plt, sol2, idxs = plot_vars)

foo = collect(zip(train_dfs, test_dfs))
y, yhat = first(foo)
data = EMA.to_data(y, mapping)
update_u0!(prob3, sys, y, mapping_ps)
fit_t = y.t
fit = EMA.global_datafit(prob3, pbounds, fit_t, data; solve_kws)
prob4 = remake(prob3; p = last.(fit), tspan = extrema(dfs[1].t))
sol4 = solve(prob4; saveat = dfs[1].t)
plt = EMA.plot_covidhub(dfs[1])
scatter!(plt, sol4, idxs = plot_vars)

fit_probs = []
fit_sols = []
(i, (y, yhat)) = first(enumerate(split_dfs))
# single_model_fits = global_ensemble_fit([prob], split_dfs, mapping;
# i = 1
# y, yhat = first(split_dfs)

global losses = []
global logged_p = []

all_losses = []
all_logged = []

for (i, (y, yhat)) in enumerate(split_dfs)
    prob = ODEProblem(sys, [], extrema(y.t))
    update_u0!(prob, sys, y, mapping_ps)
    data = EMA.to_data(y, mapping)
    global losses = []
    global logged_p = []

    fit = EMA.global_datafit(prob, pbounds, y.t, data; maxiters = 1000, solve_kws)
    push!(all_losses, losses)
    push!(all_logged, logged_p)
    fit_prob = remake(prob; p = last.(fit), tspan = extrema(dfs[i].t)) # ideally we dont call last. because not all parameters are always fit

    pkeys = parameters(sys)
    t = y.t
    train_loss = EMA.l2loss(last.(fit), (fit_prob, pkeys, t, data))
    # @show train_loss
    trts_data = EMA.to_data(dfs[i], mapping)
    trts_loss = EMA.l2loss(last.(fit), (fit_prob, pkeys, dfs[i].t, trts_data))
    @show train_loss trts_loss

    fit_sol = solve(fit_prob; saveat = dfs[i].t)
    plt = EMA.plot_covidhub(dfs[i])
    scatter!(plt, fit_sol; idxs = plot_vars)
    display(plt)
    push!(fit_probs, fit_prob)
    push!(fit_sols, fit_sol)
end

# 1. get data
# 2. load model
# 3. split data 
# 4. fit model to each data 

# an ensemble fit for [probs] [datas] becomes the cartesian product of the two arrays

# but what is the best design? i think the best thing might be that nothing to do with validation is a part of fitting

# this allows for 

# (syss .=> mappings), datas 

# sys_map_ps::AArray{Pair{ODESystem, AArray{Pair{Num, Symbol}}}}
# datas::AAray{DataFrame}
# sys->prob->sol
# [sol[num] for num in nums] 
# for 

# [prob=>mapping], [df]
