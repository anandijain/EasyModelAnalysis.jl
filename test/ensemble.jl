# @time @time_imports using EasyModelAnalysis
using EasyModelAnalysis
using DataFrames, AlgebraicPetri, Catlab
using Catlab.CategoricalAlgebra: read_json_acset
using Setfield
using MathML, JSON3
using CommonSolve

EMA = EasyModelAnalysis
datadir = joinpath(@__DIR__, "../data/")
mkpath(datadir)

# rescale data to be proportion of population
function scale_df!(df)
    for c in names(df)[2:end]
        df[!, c] = df[!, c] ./ total_pop
    end
end
function CommonSolve.solve(sys::ODESystem; prob_kws = (;), solve_kws = (;), kws...)
    solve(ODEProblem(sys; prob_kws..., kws...); solve_kws..., kws...)
end

to_ssys(sys::ODESystem) = complete(structural_simplify(sys))
to_ssys(pn) = to_ssys(ODESystem(pn))

EMA.solve(pn::AbstractPetriNet; kws...) = solve(to_ssys(pn); kws...)
getsys(sol) = sol.prob.f.sys
getsys(prob::ODEProblem) = prob.f.sys
gi(xs, y) = map(x -> x[y], xs)
cv(x) = collect(values(x))
read_replace_write(fn, rs) = write(fn, replace(read(fn, String), rs...))

# data prep
total_pop = 300_000_000
N_weeks = 20;
period_step = 10;
train_weeks = 10; # 10 weeks of training data, 10 weeks of testing

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
dfs_unscaled = EMA.select_timeperiods(orig_df, N_weeks; step = period_step)
split_dfs = [EMA.train_test_split(df; train_weeks = train_weeks) for df in dfs]
train_dfs, test_dfs = EMA.unzip(split_dfs)

dfi = dfs[1]
dftr = train_dfs[1]
dfts = test_dfs[1]

# modeling
# collection of models to treat as an ensemble
# note there is a bug in these models with a faulty parameter name "XXlambdaXX", so we string replace them
model_urls = [
    "https://github.com/indralab/mira/raw/main/notebooks/ensemble/BIOMD0000000955_miranet.json",
    "https://github.com/indralab/mira/raw/main/notebooks/ensemble/BIOMD0000000960_miranet.json",
    "https://github.com/indralab/mira/raw/main/notebooks/ensemble/BIOMD0000000983_miranet.json",
]

model_fns = [EMA.download_data(url, datadir) for url in model_urls]
map(x -> read_replace_write(x, ["XXlambdaXX" => "lambda"]), model_fns)

T_PLRN = PropertyLabelledReactionNet{Union{Number, Nothing}, Union{Number, Nothing}, Dict}
pns = read_json_acset.((T_PLRN,), model_fns)
syss = to_ssys.(ODESystem.(pns; tspan))

sys, sys2, sys3 = syss
probs = ODEProblem.(syss)
og_probs = deepcopy.(probs)
sols = solve.(probs; saveat = all_ts)
# sols = solve.(pns; tspan = (0, 10));
for sol in sols
    @test sol.retcode == ReturnCode.Success
end

psys = to_ssys(ODESystem(LabelledPetriNet(pns[1])))
@named psysb = ODESystem(ModelingToolkit.equations(psys)[1:(end - 3)])

# pns[2] debugging... 
# plotting the observed states for the custom rate law solution of pns[2] seems like its good 
# but if you plot the suseptible population, it hits a steady state and doesn't change. 
# i'm pretty sure its a model problem
lpn2 = to_ssys(ODESystem(LabelledReactionNet{Union{Number, Nothing}, Union{Number, Nothing}
                                             }(pns[2])))
lprob = ODEProblem(lpn2, collect(pns[2][:concentration]), (0, 1),
                   replace(pns[2][:rate], nothing => 0.1))
lsol = solve(lprob)
plot(lsol)

prob = probs[1]
bounds = ModelingToolkit.getbounds(getsys(prob))
ps_b, bs_ = EMA._unzip(bounds)
bounds = ps_b .=> collect.(bs_)
# bounds = ps_b .=> ((0.0, 1.0),)

# dfi = dfs[1]
# data = EMA.to_data(dfi, mapping)
# fit = EMA.global_datafit(prob, bounds, dfi.t, data; solve_kws=(;callback))
# rsol = remake_solve(prob, fit, dfi, cs_maps[1])

# logged_fits = [first.(bounds) .=> lp for lp in logged_p]

# barely improves 
remake_solve(prob, logged_fits[1], dfi, cs_maps[1])
remake_solve(prob, logged_fits[end], dfi, cs_maps[1])

fits = []
all_losses = []
all_logged = []
for dfi in dfs
    global losses = []
    global logged_p = []
    data = EMA.to_data(dfi, reverse.(col_st_map))
    fit = EMA.global_datafit(prob, bounds, dfi.t, data; solve_kws = (; callback),
                             loss = EMA.myl2loss)
    rsol = remake_solve(prob, fit, dfi, cs_maps[1])
    push!(all_losses, losses)
    push!(all_logged, logged_p)
    push!(fits, fit)
end

fit_df = fits_to_df(fits)

# this doesn't work because when we try to use the mapping to update the u0, we end up just updating an observed expr, which is meaningless 
rsol = remake_solve(probs[2], fit, dfi, reverse.(mapping))

# this is a naive way to work around the problem 
rsol = remake_solve(probs[2], fit, dfi, cs_maps[2])

@unpack Deaths, Hospitalizations, Cases, Extinct, Infected, Threatened = sys
@unpack Deceased, Infectious, Hospitalized = sys2
@unpack Infected_reported = sys3

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
fitsa = EMA.global_ensemble_fit([prob_mapping_ps[1]], train_dfs, [bounds[1]];
                                maxiters = 1000)
fitsb = EMA.global_ensemble_fit([prob_mapping_ps[2]], train_dfs, [bounds[2]];
                                maxiters = 1000)

for (i, fit) in enumerate(fitsb[1])
    remake_solve(probs[2], fit, dfs[i], cs_maps[2])
end
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

function fits_to_df(fits)
    DataFrame(namedtuple.([Symbolics.getname.(ks) .=> vs for (ks, vs) in EMA.unzip.(fits)]))
end

function logged_p_df(pkeys, logged_p)
    DataFrame(stack(logged_p)', Symbolics.getname.(pkeys))
end
# function logged_p_to_fits()

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

function remake_solve(prob, fit, df, col_st_map; tcol = :t, plot = true)
    cols, sts = EMA.unzip(col_st_map)
    ts = df[:, tcol]
    r = df[1, cols]
    new_u0 = sts .=> collect(r)
    @info new_u0

    prob = remake(prob; u0 = new_u0, p = fit, tspan = extrema(ts))
    sol = solve(prob; saveat = ts)
    plot && fit_plot(sol, df, sts)
    sol
end

"this one doesn't update u0 but uses mapping for plot"
function remake_solve2(prob, fit, df, mapping; tcol = :t, plot = true)
    sts, cols = EMA.unzip(mapping)
    ts = df[:, tcol]
    r = df[1, cols]
    prob = remake(prob; p = fit, tspan = extrema(ts))
    sol = solve(prob; saveat = ts)
    plot && fit_plot(sol, df, sts)
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

pn = pns[1]

ReactionNet{Float64, Float64}([:S => 10, :I => 1, :R => 0],
                              (:inf => 0.5) => ((1, 2) => (2, 2)),
                              (:rec => 0.1) => (2 => 3))

lrn = LabelledReactionNet{Number, Number}((:S => 0.99, :I => 0.01, :R => 0),
                                          (:inf, 0.9) => ((:S, :I) => (:I, :I)),
                                          (:rec, 0.2) => (:I => :R))

sir_sys = ODESystem(lrn)
sir_sol = solve(sir_sys; tspan = (0, 100), saveat = 1)
sir_df = DataFrame(sir_sol)
rename!(sir_df, "timestamp" => "t")
sir_bs = parameters(sir_sys) .=> ((0.0, 1.0),)
_sir_data = Symbol.(names(sir_df)[2:end]) .=> collect.(eachcol(sir_df)[2:end])
# sir_data = EMA._symbolize_args(_sir_data, EMA.sys_syms(sir_sys))

sir_prob = ODEProblem(sir_sys, [], (0, 100))

sir_data = states(sir_sys) .=> collect.(eachcol(sir_df)[2:end])
sir_fit = EMA.global_datafit(sir_prob, sir_bs, 0:100, sir_data)
sir_col_st = names(sir_df)[2:end] .=> states(sir_sys)
sir_fit_sol = remake_solve(sir_prob, sir_fit, sir_df, sir_col_st; plot = false)

lrn = LabelledReactionNet((:S => 0.99, :I => 0.01, :R => 0),
                          (:inf, 0.3 / 1000) => ((:S, :I) => (:I, :I)),
                          (:rec, 0.2) => (:I => :R))

@which ODESystem(lrn)
ODESystem(lrn)

sys = eval(LORENZ_EXPR)
@variables t x(t) y(t) z(t)
@parameters sig rho beta
@named sys2 = ODESystem(ModelingToolkit.equations(sys), t, [x, y, z], [sig, rho, beta];)

defs = Dict(sig => 10.0, rho => 28.0, beta => 8 / 3)
@set! sys2.defaults = defs

# bounds = param

p = pns[1]
sys = ODESystem(p)
ssys = to_ssys(sys)
prob = ODEProblem(ssys, [], (0, 100))
sol = solve(prob)
_obs_sts = last.(cs_maps[1])

# this is weird, eh not really. 
plt = plot(sol; idxs = _obs_sts)
plt = plot(sol; idxs = obs_sts)

dfi = dfs[1]
data = EMA.to_data(dfi, reverse.(col_st_map))
fit = EMA.global_datafit(prob, bounds, dfi.t, data; solve_kws = (; callback),
                         loss = EMA.myl2loss)
fit = EMA.global_datafit(prob, bounds, dfi.t, data)
rsol = remake_solve(prob, fit, dfi, cs_maps[1])
rsol = remake_solve(prob, fit, dfi, reverse.(mapping))

lpn = LabelledReactionNet{Number, Number}(p)
lsys = to_ssys(ODESystem(lpn))
lprob = ODEProblem(lsys, [], (0, 100))
lsol = solve(lprob)
plot(lsol)
lbounds = parameters(lsys) .=> ((0.0, 1.0),)

# this gives broken since we dont have observed on that sys dispatch
lfit = EMA.global_datafit(lprob, lbounds, dfi.t, EMA.to_data(dfi, mapping))
rsol = remake_solve(lprob, fit, dfi, cs_maps[1])

to_fit(prob) = parameters(getsys(prob)) .=> prob.p

rsol = remake_solve(lprob, to_fit(lprob), dfi, cs_maps[1])
lfit = EMA.global_datafit(lprob, lbounds, dfi.t, EMA.to_data(dfi, reverse.(cs_maps[1])))
rsol = remake_solve(lprob, fit, dfi, cs_maps[1])

lbounds = [states(lsys); parameters(lsys)] .=> ((0.0, 1.0),)

# rsol = remake_solve(lprob, to_fit(lprob), dfi, cs_maps[1])
lfit = EMA.global_datafit(lprob, lbounds, dfi.t, EMA.to_data(dfi, reverse.(cs_maps[1])))
rsol = remake_solve(lprob, lfit, dfi, cs_maps[1])
@which ODESystem(lpn)
# okay i think i've decided to throw out the covidhub fitting entirely
# i think the best mve now is to take a model, make its parameters forced states
# then perform the iterative fitting on a collection of similar models 
# this mimics the structure of the covidhub setup, without the torture of dealing with other people's models and real world data
# we probably still want to demo custom rate laws, so maybe we have the model that produces the data we fit against has custom nonlinearities 
# the other thing is resolving the inconsistent u0 updates 
Cases = Diagnosed + Recognized + Threatened
Hospitalizations = Recognized + Threatened
Deaths = Extinct
lmapping = [Cases => :cases, Deaths => :deaths, Hospitalizations => :hosp]
lmapping = [Deaths => :deaths]
# lmapping = [D(Deaths)=>:deaths] would be cool if it worked
lfit = EMA.global_datafit(lprob, lbounds, dfi.t, EMA.to_data(dfi, lmapping))

rsol = remake_solve(lprob, lfit, dfi, reverse.(lmapping))

rsol = remake_solve2(lprob, lfit, dfi, lmapping)
rlprob = remake(lprob;
                u0 = [lsys.Extinct => dfi.deaths[1], lsys.Recognized => dfi.cases[1]])
lfit = EMA.global_datafit(rlprob, lbounds, dfi.t, EMA.to_data(dfi, lmapping))
rsol = remake_solve2(rlprob, lfit, dfi, lmapping)

ldata = [Deaths => cumsum(dfi.deaths)]
lfit = EMA.global_datafit(lprob, lbounds, dfi.t, ldata)
# this one is good! 
rsol = remake_solve(lprob, lfit, dfi, reverse.(lmapping))

lbounds = parameters(lsys) .=> ((0.0, 1.0),)
lbounds = [states(lsys); parameters(lsys)] .=> ((0.0, 1.0),)

lfits = []
for dfi in dfs
    # ldata = [Deaths => cumsum(dfi.deaths)]
    ldata = [Deaths => dfi.deaths]
    # lprob = remake(lprob;
    #                 u0 = [lsys.Extinct => dfi.deaths[1], lsys.Recognized => dfi.cases[1]])
    lfit = EMA.global_datafit(lprob, lbounds, dfi.t, ldata)
    rsol = remake_solve(lprob, lfit, dfi, reverse.(lmapping);plot=false)
    plt = plot(dfi.t, dfi.deaths)
    plot!(plt, rsol; idxs=Deaths)
    display(plt)
    push!(lfits, lfit)
end

# la = Modeling

sirhd_fn = "/Users/anand/code/julia/EasyModelAnalysis.jl/data/sirhd.json"
sirhd_pn = read_json_acset(LabelledPetriNet, sirhd_fn)
sirhd = ODESystem(sirhd_pn)
defaults =EMA.sys_syms(sirhd) .=> 0.5
sirhd = ODESystem(sirhd_pn;defaults)
plot(solve(sirhd;tspan=(0,100)))
lbounds = [states(lsys); parameters(lsys)] .=> ((0.0, 1.0),)
lfit = EMA.global_datafit(lprob, lbounds, dfi.t, ldata)
