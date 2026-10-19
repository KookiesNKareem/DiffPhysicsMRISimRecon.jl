using Enzyme, KomaMRI, KomaMRIPlots, Optim, Reactant
using LinearAlgebra: dot, norm
using MRICoilSensitivities: espirit
Reactant.set_default_backend("gpu")
Reactant.allowscalar(true)

centered_axis(width, count) =
    Float32.(range(-width / 2 + width / (2count); step=width / count, length=count))

function estimate_coils(path, measured, x, y; floor=1f-2, regularization=1f-4)
    raw = RawAcquisitionData(ISMRMRDFile(path))
    names = getproperty.(raw.params["coilLabel"], :name)
    order = [only(findall(==(name), names)) for name in getproperty.(measured.params["coilLabel"], :name)]
    readout = raw.params["encodedSize"][1]
    surface = filter(p -> p.head.active_channels == length(names) && p.head.number_of_samples == readout, raw.profiles)
    body = filter(p -> p.head.active_channels == 2 && p.head.number_of_samples == readout, raw.profiles)
    raw.profiles = surface
    ky = [p.head.idx.kspace_encode_step_1 for p in raw.profiles]
    kz = [p.head.idx.kspace_encode_step_2 for p in raw.profiles]
    raw.params["encodedSize"] = [readout, maximum(ky) + 1, maximum(kz) + 1]
    calibration = AcquisitionData(raw)
    body_calibration = AcquisitionData(RawAcquisitionData(raw.params, body))
    calibration.traj[1].circular = body_calibration.traj[1].circular = false
    map_size = (readout, raw.params["reconSize"][2], raw.params["reconSize"][3])
    maps = espirit(calibration, (4, 4, 4), 16, map_size; eigThresh_1=0.02, eigThresh_2=0.0)[:, :, :, order]
    direct = Dict{Symbol,Any}(:reco => "direct", :reconSize => map_size)
    image_rss(acq) = dropdims(sqrt.(sum(abs2, reconstruction(acq, direct); dims=5)); dims=(4, 5, 6))
    surface_rss, body_rss = image_rss(calibration), image_rss(body_calibration)
    scale = surface_rss .* body_rss ./ (body_rss .^ 2 .+ regularization * maximum(body_rss)^2)
    scale ./= maximum(scale)
    adjustment = first(raw.profiles).head
    imaging = first(measured.profiles).head
    basis = hcat(Float32[adjustment.read_dir...], Float32[adjustment.phase_dir...], Float32[adjustment.slice_dir...])
    position =
        Float32[imaging.position...] .+ 1000f0 .* Float32[imaging.read_dir...] .* x' .+
        1000f0 .* Float32[imaging.phase_dir...] .* y'
    coordinates = basis' * (position .- Float32[adjustment.position...])
    axes = Tuple(centered_axis.(Float32.(raw.params["encodedFOV"]), map_size))
    receiver = ArbitraryCoilSens(axes..., maps)
    values = ComplexF32.(get_sens(receiver, coordinates[1, :], coordinates[2, :], coordinates[3, :]))
    intensity = real.(get_sens(
        ArbitraryCoilSens(axes..., reshape(ComplexF32.(scale), map_size..., 1)),
        coordinates[1, :],
        coordinates[2, :],
        coordinates[3, :]
    ))
    rss = sqrt.(sum(abs2, values; dims=2))
    intensity .* values ./ max.(rss, floor * maximum(rss))
end

function load_problem(data_directory, image_size)
    measured_path = joinpath(data_directory, "brain_gre_3t_acc/meas_MID00492_FID42184_bssfp_optimized_2x.mrd")
    adjustment_path = joinpath(data_directory, "brain_gre_3t_acc/meas_MID00478_FID42170_AdjCoilSens.mrd")
    sequence_path = joinpath(data_directory, "bssfp_slice_all_adc_optimized_R2.seq")
    sequence = resolve_triggers(read_seq(sequence_path), CardiacSignal(; heart_rate=1))
    adc = findall(block -> is_ADC_on(sequence[block]), eachindex(sequence.DUR))
    measured = RawAcquisitionData(ISMRMRDFile(measured_path))
    length(measured.profiles) == length(adc) || error("Measured profiles do not match sequence ADCs")
    target = reduce(vcat, ComplexF32.(profile.data) for profile in measured.profiles)
    fov = Float32.(measured.params["reconFOV"]) .* 1f-3
    x_axis, y_axis = centered_axis.(fov[1:2], image_size)
    x = repeat(x_axis, image_size[2])
    y = repeat(y_axis; inner=image_size[1])
    z = zeros(Float32, prod(image_size))
    coils = estimate_coils(adjustment_path, measured, x, y)
    object = Phantom(;
        x=vcat(x, x),
        y=vcat(y, y),
        z=vcat(z, z),
        ρ=ones(Float32, 2length(x)),
        T1=ones(Float32, 2length(x)),
        T2=ones(Float32, 2length(x)),
        T2s=ones(Float32, 2length(x))
    )
    receiver = vcat(coils, ComplexF32(0, 1) .* coils)
    scanner = Scanner(; receiver=KomaMRICore.CoilSensitivities(Reactant.to_rarray(receiver), nothing))
    sim = Dict{String,Any}(
        "sim_method" => Bloch(),
        "gpu" => true,
        "Nthreads" => 1,
        "return_type" => "mat",
        "precision" => "f32"
    )
    (;
        object=Reactant.to_rarray(object),
        sequence=sequence[1:last(adc)],
        scanner,
        sim,
        b=Reactant.to_rarray(target),
        scale=sum(abs2, target)
    )
end

function A(θ, p)
    n = length(θ) - 2
    object = copy(p.object)
    object.ρ .= θ[1:n]
    object.T1 .= exp.(θ[n+1:n+1])
    object.T2 .= exp.(θ[n+2:n+2])
    simulate(object, p.sequence, p.scanner; sim_params=p.sim, verbose=false)[:, :, 1]
end

loss(θ, p) = sum(abs2, A(θ, p) - p.b) / p.scale
function loss_gradient(θ, p)
    result = Enzyme.gradient(Enzyme.ReverseWithPrimal, loss, θ, Enzyme.Const(p))
    result.val, result.derivs[1]
end

function check_gradient(evaluate, θ, n; offset=1e-1, h=1e-2, tolerance=1e-2)
    direction() = vcat(norm(θ[1:n]) / sqrt(n) .* randn(n), randn(2))
    θ = θ .+ offset .* direction()
    v = h .* direction()
    G = similar(θ)
    evaluate(nothing, G, θ)
    analytic = dot(G, v)
    numeric = (evaluate(0.0, nothing, θ .+ v) - evaluate(0.0, nothing, θ .- v)) / 2
    relative = abs(analytic - numeric) / max(abs(numeric), eps())
    println("gradient check: analytic=$analytic numeric=$numeric relative_error=$relative")
    relative < tolerance || error("AD gradient check failed: $relative")
end

function reconstruct(p, image_size; warmup, iterations, output_directory)
    mkpath(output_directory)
    n = 2prod(image_size)
    device(θ) = Reactant.to_rarray(Float32.(θ))
    θ = zeros(n + 2)
    gradient = Reactant.compile(loss_gradient, (device(θ), p); sync=true)
    relaxation_fixed = Ref(true)
    function evaluate(F, G, θ)
        value, ∇ = gradient(device(θ), p)
        if G !== nothing
            copyto!(G, Array(∇))
            relaxation_fixed[] && (G[n+1:end] .= 0)
        end
        F === nothing ? nothing : Float64(Reactant.to_number(value))
    end
    rows = ["iteration,loss,T1,T2"]
    previous = fill(NaN, n + 2)
    function record(state)
        state.x == previous && return false
        copyto!(previous, state.x)
        iteration = length(rows) - 1
        t1, t2 = exp(state.x[n+1]), exp(state.x[n+2])
        push!(rows, "$iteration,$(state.f_x),$t1,$t2")
        image = reverse(reshape(abs.(complex.(state.x[1:n÷2], state.x[n÷2+1:n])), image_size); dims=1)
        savefig(
            plot_image(image; title="AD iteration $iteration"),
            joinpath(output_directory, "iteration_$(lpad(iteration, 2, '0')).png")
        )
        println("iteration=$iteration loss=$(state.f_x) T1=$t1 T2=$t2")
        false
    end
    objective = Optim.NLSolversBase.only_fg!(evaluate)
    density = optimize(objective, θ, LBFGS(m=10), Optim.Options(; iterations=warmup, callback=record))
    relaxation_fixed[] = false
    check_gradient(evaluate, Optim.minimizer(density), n)
    result = optimize(objective, Optim.minimizer(density), LBFGS(m=10), Optim.Options(; iterations, callback=record))
    println(result)
    write(joinpath(output_directory, "loss.csv"), join(rows, '\n') * "\n")
    write(joinpath(output_directory, "optim.txt"), sprint(show, result))
    θ = Optim.minimizer(result)
    complex.(θ[1:n÷2], θ[n÷2+1:n]), exp(θ[n+1]), exp(θ[n+2])
end

function run_final_ad(;
    image_size=(128, 128),
    warmup=5,
    iterations=100,
    data_directory=isempty(ARGS) ?
        joinpath(homedir(), "Desktop/Data/cleaner brain data acq 21 august") : first(ARGS),
    output_directory=joinpath(@__DIR__, "FINAL_AD")
)
    reconstruct(load_problem(data_directory, image_size), image_size; warmup, iterations, output_directory)
end

abspath(PROGRAM_FILE) == (@__FILE__) && run_final_ad()
