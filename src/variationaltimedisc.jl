module VariationalTimeDisc

using LinearAlgebra
using Printf
using Jacobi

struct TimeDiscretization{T, Ti}
    kL::Ti
    kR::Ti
    opt::Array{Ti, 1}
    super::Array{Ti, 1}
    p::Array{T, 1}
    MassCoeffs::Array{T, 2}
    StiffCoeffs::Array{T, 2}
    nQF::Ti
    pQF::Array{T, 1}
    wQF::Array{T, 1}
    IC::Array{T,1}
    KAPW::Array{T,2}
    KA1PW::Array{T,2}
    # parameters for post processing
    pp::Ti
    CorrectionPW::Array{T,2}
    CorrectionDerivPW::Array{T,2}
    DerivL::Array{T,2}
    DerivR::Array{T,2}
    CorrectionL::Array{T,2}
    CorrectionR::Array{T,2}
end

#=
Convolution of two vectors
=#
function Convolution(u::Array{T,1},v::Array{T,1}) where T
    lu = length(u)
    lv = length(v)
    
    s=zeros(T,lu+lv-1)

    for i = 1: lu+lv-1
        t=zero(T)
        for j=max(1,i+1-lv):min(i,lu)
            t=t+u[j]*v[i-j+1]
        end
        s[i] = t
    end
    return s
end

#=
 Generating all data needed for Variational Time Discretization
=#
function SetVariationalTimeDisc(T, R, K)
    # R: polynomial degree
    # K = 0; dG method; K == 2, post-processed dG 
    # K = 1; cGP method;K == 3, post-processed cGP
    if K>R
        error("Variational TimeDiscretization is implemented only for 0≤k≤r")
    end 
    r = R
    k = K

    opt = [r+1, r, 2*r+1-k, 2*r+1-k, 2*r+1-k, 2*r+1-k]

    if k <= 1
        opt[4] = r 
        opt[6] = r
    end

    if k<r
        super = [r+2, r+1, 2*r+1-k, 2*r+1-k, 2*r+1-k, 2*r+1-k]
    else
        super = copy(opt)
        super[2] = r+1
    end

    # number of values (fct + derivative) at left and right end reduced by 1: number of collocation conditions
    kL = floor(Int, (k+1)/2)
    kR = floor(Int, (k+2)/2)
    pp = floor(Int, (k+1)/2)

    println()
    print("VariationalTimeDisc(r=", r, ", k=", k, "): ")

    if iseven(k)
        if k==0
            println("dG(", r, ")")
        else
            println("dG-C(", kL-1,")(", r, ")")
        end
    else
        if k==1
            println("cGP(", r, ")")
        else
            println("cGP-C(", kL-1, ")(", r, ")")
        end
    end

    # w = zgj(r-k, kL, kR, T)
    w = zgj(r-k, kR, kL, T)
    # vector with all inner quadrature nodes and +1
    p = [w; 1]
    # get basis for ansatz space using interpolation 
    MA = zeros(T, r+1, r+1)

    # left end point 
    for i=1:kL 
        for j=i:r+1 
            MA[j, i] = T((-1)^(j-1-i+1) * factorial(BigInt(j-1))//factorial(BigInt(j-1-i+1)))
        end
    end
    # inner points 
    for i= 1 : r+1-kR-kL
        for j = 1 : r+1
            MA[j, i+kL] = p[i]^(j-1)
        end
    end
    # right end point 
    for i=1:kR 
        for j=i:r+1 
            MA[j, r+1-kR+i] = T(1^(j-1-i+1) * factorial(BigInt(j-1))//factorial(BigInt(j-1-i+1)))
        end
    end

    KA = inv(MA)
    
    # test basis, lagrange basis for the inner points and +1
    MT = zeros(T, r+1-k, r+1-k)
    for i = 1 : r+1-k 
        for j= 1 : r+1-k 
            MT[j, i] = p[i]^(j-1)
        end
    end
    KT = inv(MT)   

    MassCoeffs  = zeros(T, r-k+1, r+1)
    StiffCoeffs = zeros(T, r-k+1, r+1)
    N = zeros(T, 2*r+1-k)
    N[1:2:(2*r+1-k)] = T.(2 .// (1:2:(2*r+1-k)))
    NL = ones(T, 2*r+1-k)
    NL[2:2:(2*r+1-k)] .= -one(T)
    
    NTL = ones(T, r+1-k)
    NTL[2:2:(r+1-k)] .= -one(T)

    # Determine monomial coefficients for derivative of BF
    KA1 = zeros(T, r+1, r+1)
    for i = 1 : r 
        KA1[:, i] = KA[:, i+1] * i
    end

    if k==0
        IC = zeros(T, r+1)
    else
        IC = []
    end
    
    for i = 1 : r-k+1
        for j = 1 : r+1
            W = Convolution(KA[j,:], KT[i,:])
            StiffCoeffs[i, j] = W' * N
    
            if k == 0
                MassCoeffs[i, j] = W' * NL
            end
            W = Convolution(KA1[j, :], KT[i, :])
            MassCoeffs[i, j] = MassCoeffs[i, j] + W' * N
        end
        if k==0
            IC[i] = KT[i, :]' * NTL
        end
    end
    # Bases change for the test functions, now: A-share to inner QF points and +1 becomes unit matrix.
    
    KT = StiffCoeffs[1:(r+1-k),(1+kL):(r+1+kL-k)] \ KT    
       
    MassCoeffs[:] .= zero(T)
    StiffCoeffs[:] .= zero(T)

    for i = 1 : r-k+1
        for j = 1 : r+1 
            W = Convolution(KA[j, :], KT[i, :])
            StiffCoeffs[i, j] = W' * N

            if k == 0
                MassCoeffs[i, j] = W' * NL
            end

            W = Convolution(KA1[j, :], KT[i, :])
            MassCoeffs[i, j] = MassCoeffs[i, j] + W' * N
        end
        if k == 0
            IC[i] = KT[i, :]' * NTL
        end
    end

    nQF = r + 4
    pQF = zeros(T, nQF+1)
    pQF[1:nQF] = zgj(nQF, 0, 0, T)
    pQF[nQF+1] = one(T)
    v = zeros(T, nQF)
    v[1:2:nQF] = T.(2 .// (1:2:nQF))
    # get weights by solving a linear system
    MM = zeros(T,nQF,nQF)

    for i=1:nQF
        for j=1:nQF
            MM[j,i] = pQF[i]^(j-1)
        end
    end
    wQF = MM \ v

    PW = zeros(T,2*r+2-k,nQF+1)
    for i=1:2*r+2-k
        for j=1:nQF+1
            PW[i,j] = pQF[j]^(i-1)
        end
    end

    # KAPW[i,j] = Value of the i-th BF at the j-th QF point
    KAPW = KA*PW[1:(r+1),:]

    KA1PW = KA1*PW[1:(r+1),:]

    ## data preparation for post processing
    if r >= k 
      Correction   = zeros(T, r-k+1, 2*r+2-k)
      CorrectionD  = zeros(T, r-k+1, 2*r+2-k)
      CorrectionPP = zeros(T, r-k+1, 2*r+2-k)
      for j = 0 : r-k
        correctionpp = [one(T)]
        for i = 1 : (kL + j)
          correctionpp = Convolution(correctionpp, [one(T), one(T)])
        end
        for i = 1 : (kR + j)
          correctionpp = Convolution(correctionpp, [-one(T), one(T)])
        end
        # zeros of Jacobi polynomial
        gp = zgj(r-k-j, kR+j, kL+j, T)
        for i = 1 : r-k-j
          correctionpp = Convolution(correctionpp, [-gp[i], one(T)])
        end

        corr = copy(correctionpp)
        for kk = 1:pp + j
          for i = 1 : r+1+j 
            correctionpp[i] = correctionpp[i+1]*i
          end
          correctionpp[r+2+j] = zero(T)
        end
        # Horner scheme for pth derivative in -1
        s = correctionpp[r+2+j]
        for kk = r+1+j : -1: 1
          s = (-1) * s + correctionpp[kk]
        end
        corr = corr/s
        correctionpp = correctionpp/s

        Correction[j+1, 1:(r+2+j)] = corr
        CorrectionPP[j+1, 1:(r+2+j)] = correctionpp

        for i = 1 : r + 1 + j
          corr[i] = corr[i+1]*i
        end
        corr[(r+2+j):end] .= zero(T)
        CorrectionD[j+1,1:(r+2+j)] = corr
      end
    else
        CorrectionPW = zeros(T, 0, 0)
        CorrectionDerivPW = zeros(T, 0, 0)
        DerivL = zeros(T, 0, 0)
        DerivR = zeros(T, 0, 0)
        CorrectionL = zeros(T,0,0)
        CorrectionR = zeros(T,0,0)
    end
    if r >= k
      CorrectionPW = Correction * PW
      CorrectionDerivPW = CorrectionD * PW
    end
    # post processing for left and right points
    if r >= k
      DerivL = zeros(T, r+1, r-k+1)
      DerivR = zeros(T, r+1, r-k+1)
      CorrectionL = zeros(T, r+1-k, r+1-k)
      CorrectionR = zeros(T, r+1-k, r+1-k)
      #pp-th derivative of base and correction function at -1 and +1
      KApp = copy(KA)
      LR = ones(T, 2*r+2-k, 2)
      LR[2:2:end,1] .= -one(T)
      for j = 1 : pp 
        for i = 1 : r
          KApp[:, i] = KApp[:, i+1] * i
        end
        KApp[:, r+2-j] .= zeros(T)
      end
      DerivL[:,1] = KApp*LR[1:(r+1),1]
      DerivR[:,1] = KApp*LR[1:(r+1),2]
      for j = 1 : r-k 
        for i = 1 : r 
          KApp[:, i] = KApp[:, i+1] * i
        end
        KApp[:, r+2-(pp+j)] .= zeros(T)
        DerivL[:, 1+j] = (KApp * LR[1:(r+1), 1])
        DerivR[:, 1+j] = (KApp * LR[1:(r+1), 2])
      end
      # correction for left and right end points
      for j = 1 : r-k+1 
        temp = CorrectionPP[j, :]
        for jj = 1 : 2+r-k-j
           CorrectionR[j, jj] = temp' * LR[:, 2]
           for i = 1 : 2*r - k 
            temp[i] = temp[i+1] * i
           end
           temp[2*r+1-k] = zero(T)
           CorrectionL[j, jj] = temp' * LR[:, 1]
        end
      end
    end
    ##

    return TimeDiscretization{T, Int}(kL, kR, opt, super, p, MassCoeffs, 
                StiffCoeffs, nQF, pQF, wQF, IC, KAPW, KA1PW, pp, CorrectionPW, 
                CorrectionDerivPW, DerivL, DerivR, CorrectionL, CorrectionR)
# SetVariationalTimeDisc
end

# VariationalTimeDisc
end
