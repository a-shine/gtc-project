using DifferentialEquations
using DelimitedFiles
using Random

#####################################
### Deterministic part of the SDE ###
#####################################
function RS2_compaction(dstate, state, p, t)
	x, y, z, u = state
	κ, β₁, β₂, ρ, λ, ν, α, γ, ϵ₂, ϵ₄ = p
    expx = exp(x)
    dstate[3] = - ρ*(β₂*x + z)*expx
    dstate[4] = - α - γ*u + dstate[3]
	dstate[1] = (expx*((β₁-1)*x*(1+λ*u) + y - u) + κ*(1-expx) - dstate[4]*(1+λ*y)/(1+λ*u)) / (1 + λ*u + ν*expx)
    dstate[2] = κ*(1 - expx) - ν*expx*dstate[1]
end

##################################
### Stochastic part of the SDE ###
##################################
function σ_RS2_compaction(dstate,state,p,t)
    κ, β₁, β₂, ρ, λ, ν, α, γ, ϵ₂, ϵ₄ = p
    dstate[1] = 0.0
    dstate[2] = ϵ₂
    dstate[3] = 0.0
    dstate[4] = ϵ₄
end

#######################################
### Parameters independent from σₙ₀ ###
#######################################

# Loading velocity v₀ and reference velocity vstar are used interchangeably
v₀ = 10e-6       # Loading velocity [m/s]

k  = 14.8*1e9    # Spring stiffness [Pa/m]

β₁ = 1.2         # b₁/a₀ [non-dim]

L₁ = 3*1e-6      # Critical distance for θ₁ [m]
ρ = 0.1
L₂ = ρ*L₁

a = 0.01         # Friction direct effect [non-dim]
μ₀ = 0.64        # Static friction coefficient [non-dim]
λ = a/μ₀

G = 31e9         # Rigidity modulus of quartz [Pa]
ρᵥ = 2.65e3      # Density of quartz [kg/m^3]
cₛ = sqrt(G/ρᵥ)
η₀ = G/(2*cₛ)
η = 15*η₀ # drift: deterministic chaos
#η = 20*η₀ # drift: limit cycle

p₀ = 1.01325e5   # Reference surrounding pressure (atmospheric pressure) [Pa]

βₐ = 1e-9        #[0.5-4]*1e-9 (David et al., 1994; see Segall and Rice, 1995)
βₘ = 1e-11       # Compressibility of Quartz (Pimienta et al., 2017, JGR, fig. 12)
ϕ₀ = 0.075       # Reference porosity
β = ϕ₀*(βₐ+βₘ)
ϵ = -0.017*1e-3  # Dilatancy/Compressibility coefficient

c₀ = 0.1          # Diffusivity [1/s]
γ = c₀*L₁/v₀

ϵₒₜ = 0.004*1e6  # Standard deviation on frictional shear stress τ [Pa]
ϵₒₛ = 0.006*1e6  # Standard deviation on normal stress σₙ [Pa]

# Noise level to perturb the dynamics on the frictional shear stress τ [Pa]
ϵₜ = 0.0 * ϵₒₜ   # ordinary differential equations
#ϵₜ = 0.5 * ϵₒₜ   # stochastic differential equations
# Noise level to perturb the dynamics on the normal stress σₙ [Pa]
ϵₛ = 0.0 * ϵₒₛ   # ordinary differential equations
#ϵₛ = 0.5 * ϵₒₛ   # stochastic differential equations

##########################
### Output directories ###
##########################
dir_main = pwd() * "/"
dir_data = dir_main * "../data/synquakes/Gualandietal2023/"

#####################
### Studied cases ###
#####################
# Reference normal stresses σₙ₀ [Pa] and experiment names
σₙ₀ = Vector{Float64}(undef, 3)
exp_names = Vector{String}(undef, 3) 
σₙ₀[1] = 15.001*1e6; exp_names[1]  = "b726"
σₙ₀[2] = 17.379*1e6; exp_names[2]  = "b698"
σₙ₀[3] = 24.976*1e6; exp_names[3] = "i417"
n_exp = size(exp_names)[1]

##############################
### Integration parameters ###
##############################
sampling_rate = 0.01
maxiters = 1e8
tin  = 0.0
tfin = 8000
x₀ = 0.05
y₀ = 0.0
z₀ = 0.0
u₀ = 0.0
state₀ = [x₀, y₀, z₀, u₀]

##################################
### Loop to simulate each case ###
##################################
for ii = 1:n_exp
    # For reproducibility set the random seed for each case
    Random.seed!(42)
    print("\n")
    print(ii)
    #####################################
    ### Parameters dependent from σₙ₀ ###
    #####################################
    α = (c₀*p₀*L₁) / (v₀*λ*σₙ₀[ii])
    τ₀ = μ₀*σₙ₀[ii]
    β₂ = -ϵ/(λ*β*σₙ₀[ii])
    κ = (k*L₁) / (a*σₙ₀[ii])
    ν = η*v₀ / (a*σₙ₀[ii])
    ϵ₂ = ϵₜ / (a*σₙ₀[ii])
    ϵ₄ = (1/λ)*ϵₛ/σₙ₀[ii]
    print(" σₙ₀ = ")
    print(σₙ₀[ii]/1e6)
    print(" MPa     ")
    print(" κ/κ₀ = ")
    print(κ/(β₁-1))
    # 8 parameters + 2 for the noise
    # (the 9th parameter vstar is equal to v₀, so the ratio v₀ / vstar = 1
    # and is already taken into account in the way the ODE/SDE are written)
    p = (κ, β₁, β₂, ρ, λ, ν, α, γ, ϵ₂, ϵ₄)

    ##################
    ### Simulation ###
    ##################
    prob_sde_RS2_compaction = SDEProblem(RS2_compaction,σ_RS2_compaction,state₀,(tin,tfin),p)
    sol = solve(prob_sde_RS2_compaction, saveat=sampling_rate*v₀/L₁,maxiters=maxiters)
    
    ####################
    ### Save results ###
    ####################
    writedlm(string(dir_data, exp_names[ii],"/sol_t.txt"), sol.t)
    writedlm(string(dir_data, exp_names[ii],"/sol_u.txt"), sol.u)
    writedlm(string(dir_data, exp_names[ii],"/a.txt"), a)
    writedlm(string(dir_data, exp_names[ii],"/sigman0.txt"), σₙ₀[ii])
    writedlm(string(dir_data, exp_names[ii],"/L1.txt"), L₁)
    writedlm(string(dir_data, exp_names[ii],"/v0.txt"), v₀)
    writedlm(string(dir_data, exp_names[ii],"/tau0.txt"), τ₀)
    writedlm(string(dir_data, exp_names[ii],"/p.txt"), p)
end