module Example222_ReactionDiffusionVTD
    include("../src/variationaltimedisc.jl")    
    using LinearAlgebra
    using Printf
    using Jacobi

    using ExtendableFEM
    using ExtendableGrids
    using GridVisualize    
    using ExtendableSparse
    using Symbolics

    ## parameters
    const α = 1
    const β = [1, 1]
    const μ = 1
    
    ## exact data for problem by Symbolics
    function prepare_data()

        @variables x y t

        ## exact function
        u = (1+t)*x

        ## gradient of u
        ∇u = Symbolics.gradient(u, [x, y])

        ## Laplacian
        Δu = Symbolics.gradient(∇u[1], [x]) + Symbolics.gradient(∇u[2], [y])

        ## right-hand side
        f = Symbolics.gradient(u, [t])[1] - μ * Δu[1] + α * u + dot(β, ∇u)

        ## build functions
        u_eval = build_function(u, x, y, t, expression = Val{false})
        ∇u_eval = build_function(∇u, x, y, t, expression = Val{false})
        f_eval = build_function(f, x, y, t, expression = Val{false})

        ## for vector-valued functions build functions creates an array of functions: 
        ## ∇u_eval[1](x,y,t) = result
        ## ∇u_eval[2](result, x, y, t)
        return f_eval, u_eval, ∇u_eval[2]
    end

    function operator_kernel!(result, input, qpinfo)
        u, ∇u = view(input, 1), view(input, 2:3)
        # α * u_h + β_1 * ∇_xu_h + β_2 ∇_y u_h
        result[1] = α * u[1] + dot(β, ∇u)
        result[2:3] = μ * ∇u
        return nothing
    end
    
    function main(;time_order = 0, scheme = 0, nlevels=2, bonus_quadorder = 2,
        T0 = 0, end_time=1, nsteps=10, space_order = 2, Plotter = nothing, kwargs...)

        f_eval, u_eval, ∇u_eval = prepare_data()

        f! = (result, qpinfo) -> (result[1] = f_eval(qpinfo.x[1], qpinfo.x[2], qpinfo.time))
        u! = (result, qpinfo) -> (result[1] = u_eval(qpinfo.x[1], qpinfo.x[2], qpinfo.time))
        ∇u! = (result, qpinfo) -> (∇u_eval(result, qpinfo.x[1], qpinfo.x[2], qpinfo.time))

        ## prepare error calculation
        function exact_error!(result, u, qpinfo)
            u!(result, qpinfo)
            ∇u!(view(result, 2:3), qpinfo)
            result .-= u
            result .= result .^ 2
        end
        ErrorIntegrator = ItemIntegrator(exact_error!, [id(1), grad(1)]; quadorder = 2*space_order, kwargs...)

        ## creating operators
        pde_operator = BilinearOperator(operator_kernel!, [id(1), grad(1)])
        rhs_operator = LinearOperator(f!, [id(1)]; bonus_quadorder = bonus_quadorder)
        bnd_operator = InterpolateBoundaryData(1, u!; regions = 1:4)

        # time_order : polynomial order in time 
        # scheme : 0 (dG scheme)
        # scheme : 1 (cGP scheme)
        TDisc = VariationalTimeDisc.SetVariationalTimeDisc(Float64,time_order, scheme)   

        # inital grid 
        xgrid = uniform_refine(grid_unitsquare(Triangle2D), nlevels)

        # choose a finite element type
        FEType = H1Pk{1,2,space_order}
        FES = FESpace{FEType}(xgrid)
        sol = FEVector(FES)

        n_dofs = FES.ndofs
        interpolate!(sol[1], u!; time = 0.0)
        initSol = zeros(Float64, n_dofs)
        initSol = sol.entries

        ## mass matrix
        M = FEMatrix(FES)
        assemble!(M, BilinearOperator([id(1)]))
        
        ## stiffness matrix
        A = FEMatrix(FES)
        assemble!(A, pde_operator)

        ## right-hand side
        rhs = FEVector(FES)
        assemble!(rhs, rhs_operator)


        ## plot initial solution
	    println(stdout, unicode_scalarplot(sol[1]; title = "u (init)", kwargs...))
	    #pl = GridVisualizer(; Plotter = Plotter, layout = (2, 2), clear = true, size = (1000, 1000))
        #scalarplot!(pl[1,1], sol[1])

        ## boundary data
        assemble!(bnd_operator, FES; time = 0.0)
        bdofs = fixed_dofs(bnd_operator)
        bddata = fixed_vals(bnd_operator)
        for dof in bdofs
            A.entries[dof, dof] = 1e60
            rhs.entries[dof] = 1e60 * bddata[dof]
        end

        #=
        declarations for the time Discretization 
        =#
        r = time_order
        k = scheme
        kL = TDisc.kL
        kR = TDisc.kR
        p = TDisc.p
        MassCoeffs = TDisc.MassCoeffs
        StiffCoeffs = TDisc.StiffCoeffs
        IC = TDisc.IC
        KAPW = TDisc.KAPW
        KA1PW = TDisc.KA1PW

        nQF = TDisc.nQF
        pQF = TDisc.pQF
        wQF = TDisc.wQF

        t0 = T0
        tau = (end_time - T0)/nsteps
        tend = t0 + tau
        StiffMatArray = Array{FEMatrix{Float64},1}(undef, kR)
        RhsArray = Array{FEVector{Float64}, 1}(undef, r)
        V1 = zeros(Float64, FES.ndofs, r+1)

        if k>0
            Mu0 = zeros(Float64, FES.ndofs, kL)
        else
            Mu0 = zeros(Float64, FES.ndofs)
        end
        
        # SystemMatrix = FEMatrix([FES, FES])
        SysFES = Array{FESpace{Float64, Int32}, 1}([])
        for j=1:r+1-k
            push!(SysFES, FES)
        end
        SystemMatrix = FEMatrix(SysFES)
        # @show SystemMatrix
        SystemRHS = FEVector(SysFES)
        SystemSol = FEVector(SysFES)
        
        # number of solution vectors to be calculated
        d = r+1-kL
        # Number of inner solution vectors
        di = d-kR


        discsol = zeros(Float64,n_dofs,r+1)
        discsolPW = zeros(Float64,n_dofs,nQF+1)

        eL2 = zero(Float64)
        errorL2 = zeros(Float64, nQF)
        eH1 = zero(Float64)
        errorH1 = zeros(Float64, nQF)

        error = evaluate(ErrorIntegrator, sol; time = 0.0)
        l2 = sqrt(sum(view(error, 1, :)))
        h1 = sqrt(sum(view(error, 2, :)) + sum(view(error, 3, :)))
        println("L2 error init: ", sqrt(l2))
        println("H1 error init: ", sqrt(h1))

        for current_time = 1 : nsteps 
            @printf("Time step: %d: [%.5f, %.5f]\n", current_time, 
            t0, t0+tau)
        
            if kL > 0 # cGP
                V1[:, 1] = rhs.entries                
            end
            # inner quad points
            for i = 1 : r - k
                fill!(rhs.entries, 0)               
                assemble!(rhs, rhs_operator; time = t0 + tau * (p[i]+1)/2)
                V1[:,i + kL] = rhs.entries
                # println(rhs[1])
            end
            fill!(rhs.entries, 0)
            assemble!(rhs, rhs_operator; time = t0 + tau)
            # println(rhs[1])
            V1[:, 1+(r+1-kR)] = rhs.entries
            
            #
            # println("V1: ", V1)
            Mu0[:] = M.entries*sol[1].entries
            # println(typeof(Mu0))

            fill!(SystemRHS.entries, 0)
            for i = 1 : r+1-k
                # println(size(MassCoeffs[i, 1:kL]))
                if k>0
                    addblock!(SystemRHS[i], Mu0 * MassCoeffs[i, 1:kL]; factor= - 1.0)

                    addblock!(SystemRHS[i], A.entries * sol[1].entries; factor= - tau/2 * StiffCoeffs[i, 1])
                else
                    addblock!(SystemRHS[i], Mu0; factor= IC[i])
                end
            end

            
            for i= 1 : r+1-k
                addblock!(SystemRHS[i], V1 * StiffCoeffs[i, :]; factor= tau/2 )
            end
            

            # reset the system matrix
            fill!(SystemMatrix.entries.cscmatrix.nzval, 0)

            for s1 = 1 : di + 1
                fill!(A.entries.cscmatrix.nzval, 0)
                assemble!(A, pde_operator; time = t0 + tau * (p[s1] +1) / 2. )
                for s2 = 1 : r + 1 - k
                    addblock!(SystemMatrix[s2, s1], M[1, 1]; factor= MassCoeffs[s2, s1+kL])
                    addblock!(SystemMatrix[s2, s1], A[1, 1]; factor= StiffCoeffs[s2, s1+kL] * tau / 2)
                end
            end

            for i = 1 : r+1-k
                assemble!(bnd_operator; time = t0 + tau * (p[i]+1)/2)

                for dof in bdofs                    
                    SystemRHS[i][dof] = 1e60 * bddata[dof]
                    SystemMatrix[i,i][dof,dof] = 1e60
                end
            end
            
            flush!(SystemMatrix.entries)            

            SystemSol.entries[:] = SystemMatrix.entries \ SystemRHS.entries

            # interpolate!(sol[1], u, time=t0+tau)
            # defect = SystemMatrix.entries * sol[1].entries - SystemRHS.entries
            # println( "defect :  ", defect)

            # for j = 1 : length(sol.entries)
            #     sol[1][j] = SystemSol[di+1][j]
            # end

            # interpolate!(sol[1], u; time = t0 +tau)
            # gridplot!(pp[2,1], xgrid; linewidth = 1)
            # scalarplot!(pp[1,1], xgrid, nodevalues_view(sol[1])[1], 
            # levels = 4, title = "u_h")
            # error("stops")

            # error computation 
            if k>0
                discsol[:] = [initSol reshape(SystemSol.entries, n_dofs, d)]
            else
                discsol[:] = reshape(SystemSol.entries, n_dofs, d)
            end
            discsolPW[:] = discsol*KAPW

            for i = 1 : nQF
                ti = t0+tau*(pQF[i]+1)/2
                error = evaluate(ErrorIntegrator, sol; time = ti)

                for j = 1 : n_dofs
                    sol[1][j] = discsolPW[j, i]
                end

                l2 = sqrt(sum(view(error, 1, :)))
                h1 = sqrt(sum(view(error, 2, :)) + sum(view(error, 3, :)))
                errorL2[i] = l2
                errorH1[i] = h1
            end
            eL2 = eL2 + errorL2' * wQF * tau/2
            eH1 = eH1 + errorH1' * wQF * tau/2

            # error computation at discrete time points
            for j = 1 : n_dofs
                sol[1][j] = discsolPW[j, nQF+1]
            end

            error = evaluate(ErrorIntegrator, sol; time = t0+tau)

            l2 = sqrt(sum(view(error, 1, :)))
            h1 = sqrt(sum(view(error, 2, :)) + sum(view(error, 3, :)))
            println("L2 error: ", sqrt(l2))
            println("H1 error: ", sqrt(h1))


            ###############################
            # discsolPW[:] = discsol*KA1PW / (tau/2)

            # for i = 1 : nQF
            #     H1Error = L2ErrorIntegrator(∇(u), Gradient; time= t0+tau*(pQF[i]+1)/2 )
            #     for j = 1 : n_dofs
            #         sol[1][j] = discsolPW[j, i]
            #     end

            #     h1 = evaluate(H1Error,   sol[1])
            #     
            #     # println(l2)
            # end
            # eH1 = eH1 + errorH1' * wQF * tau/2

            # H1Error = L2ErrorIntegrator(∇(u), Gradient; time= t0+tau )

            # for j = 1 : n_dofs
            #         sol[1][j] = discsolPW[j, nQF+1]
            # end

            # h1 = evaluate(H1Error, sol[1])
            # println("H1 error: ", sqrt(h1))
            ##############################

            # copy sol for the next time step
            for j = 1 : length(sol.entries)
                sol[1][j] = SystemSol[di+1][j]
            end
            # sol[1] .= SystemSol[di+1]
            # @show SystemMatrix

            t0 = t0 + tau
            
        end # endfor time loop
        println(sqrt(eL2))
        println(sqrt(eH1))
	    println(stdout, unicode_scalarplot(sol[1]; title = "u (final)", kwargs...))
        # @show rhs.entries
        # prepare the system matrix
    end # end Main function
end