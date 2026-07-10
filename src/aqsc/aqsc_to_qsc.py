import aqsc
import numpy as np
import jax.numpy as jnp

def aqsc_to_qsc(eq, nphi=None):
    try: 
        from scipy.constants import mu_0
        from qsc import Qsc
    except:
        raise ImportError('pyQSC must be installed to convert an aqsc equilibrium to a qsc equilibrium.')
    if nphi is None:
        nphi = eq.tau.content.shape[1]
    B_denom0 = eq.B_denom[0]
    B0 = np.sqrt(np.real(1/B_denom0))

    ''' p2 '''
    # 2πψ=πr^2B_ref
    # 2ψ/B_ref=r^2
    # r = sqrt(2ψ/B_ref)
    # Since eps = sqrt(ψ)
    # r = sqrt(2/B_ref) eps
    # d^2p/dr^2 = d^2p/(2/B_ref)deps^2 
    p2_eps = jnp.real(jnp.average(eq.p_perp[2][0].content))
    p2_r = p2_eps * B0 / 2

    ''' B1 and B2 (not to be confused with B_denom 1 and 2)'''
    # B = B_denom^-1/2
    # B' = (-1/2)B_denom^-3/2 B_denom'
    # B'' = (-1/2)(-3/2)B_denom^-5/2 (B_denom')^2 + (-1/2)B_denom^-3/2 B_denom''
    #     =         3/4 B_denom^-5/2 (B_denom')^2 + (-1/2)B_denom^-3/2 B_denom''
    # Substitute in eps = 0:
    B1_eps = (-1/2) * B_denom0 ** (-3/2) * eq.B_denom.deps()[0]
    B1_r = B1_eps * np.sqrt(B0 / 2)
    B1s_r = np.real(B1_r.exp_to_trig().content[0, 0]) # Take the c1 component of B1 (not to be confused with B_denom[1])
    B1c_r = np.real(B1_r.exp_to_trig().content[1, 0]) # Take the c1 component of B1 (not to be confused with B_denom[1])
    B2_eps = (
        3/4 * B_denom0 ** (-5/2) * (eq.B_denom.deps() * eq.B_denom.deps())[0]
        -1/2 * B_denom0 ** (-3/2) * eq.B_denom.deps().deps()[0]
    ) / 2 # The factor of 2 comes from eps^2
    B2_r = B2_eps * B0 / 2
    # B20 is not a degree of freedom.
    B2s_r = np.real(B2_r.exp_to_trig().content[0, 0])
    B2c_r = np.real(B2_r.exp_to_trig().content[2, 0])

    ''' I and G '''
    # I = ∫B_theta dtheta
    # G = ∫B_phi dphi
    # Integrating theta under a fixed phi is the same as integrating chi
    I2 = np.real(np.average(eq.B_theta[2].antid_chi().content))
    # B_phi_0 = B_alpha[0] - iota_bar[0] * B_theta[0]
    # But since B_theta[0] = 0, the second term vanishes
    # and B_phi0 is a constant. Therefore, its phi integral
    # must also be a constant
    B_phi_0 = eq.B_alpha[0] #  - eq.iota*eq.B_theta
    sG = np.sign(B_phi_0)

    return Qsc(
        rc=eq.axis_info['rc'], # checked rc: the cosine components of the axis radial component
        zs=eq.axis_info['zs'], # checked zs: the sine components of the axis vertical component
        rs=eq.axis_info['rs'], # checked rs: the sine components of the axis radial component
        zc=eq.axis_info['zc'], # checked zc: the cosine components of the axis vertical component
        nfp=eq.nfp, # checked nfp: the number of field periods
        # Looks like pyqsc and aqsc are phase shifted
        # since stellsym configs in aqsc has B2s_r = 0.
        etabar=B1c_r/B0, # B1c_r/B0, # etabar: B1c/B0 a scalar that specifies the strength of the first order magnetic field modulus
        sigma0=eq.sigma0, # sigma0: the value of the function sigma at phi=0, which is 0 for stellarator-symmetry
        B0=B0, # checked B0: the strength of the magnetic field on-axis
        I2=I2, # checked (only for nearly-current-free cases) I2: the second derivative of the current with respect to the radial variable r
        sG=sG, # checked sG: sign of the Boozer function G
        spsi=1, # checked spsi: sign of the toroidal flux function Psi
        nphi=nphi, # checked nphi: toroidal resolution specifying the number of points in a grid along the axis
        B2s=B2s_r, # B2s: a scalar that specifies the strength of the sine component of the second order magnetic field modulus, 0 for stellarator-symmetry
        B2c=B2c_r, # B2c: a scalar that specifies the strength of the cosine component of the second order magnetic field modulus
        # PyAQSC seems to use p without mu0. See Constructing ... high order eq 2.7.
        p2=p2_r, # p2: the second derivative of the pressure with respect to the radial variable r, usually negative
        order='r3', # order: a string that specifies the order of the expansion, “r1”, “r2” or “r3”. For “r3” only the X3c1, Y3c1 and Y3s1 components are calculated (see section 3 of [LandremanSengupta2019])
    )