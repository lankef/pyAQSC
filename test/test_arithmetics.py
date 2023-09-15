import unittest
import numpy as np
from aqsc import *

import jax.numpy as jnp

# Size of the chi and phi grid used for evaluation tests
points = np.linspace(0, 2*np.pi*(1-1/n_grid_phi), n_grid_phi)
chi = np.linspace(0, 2*np.pi*(1-1/n_grid_chi), n_grid_chi)
phi = points
psi = np.linspace(0,5,100)

# The numerical derivatives usually aren't as accurate 
# as to go below the threshold of np.isclose().

class TestArithmetics(unittest.TestCase):

    def test_operator_logic(self):
        '''
        Testing logics in +, -, *, / for zero-handling 
        and error throwing.
        '''
        a = jnp.array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [4,5,6],
            [1,2,3],
        ])

        a2 = jnp.array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [4,5,6],
        ])

        b = jnp.array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
        ])

        b2 = jnp.array([
            [1,2,3],
            [4,5,6],
        ])

        c = jnp.array([
            [10],
            [20],
            [30],
        ])

        r = jnp.array([
            [10,100,1000],
        ])

        g = jnp.array([
            [10,100,1000,10000],
        ])

        test_even_a = ChiPhiFunc(a, 4)
        test_odd_a = ChiPhiFunc(a2, 4)
        test_even_b = ChiPhiFunc(b, 4)
        test_odd_b = ChiPhiFunc(b2, 4)
        single_row = ChiPhiFunc(r, 4)
        single_col = ChiPhiFunc(c, 4)
        wrong_grid = ChiPhiFunc(g, 4)
        test_odd_a_wrong_nfp = ChiPhiFunc(a, 2)
        zero = ChiPhiFuncSpecial(0)
        null_a = ChiPhiFuncSpecial(-2)
        null_b = ChiPhiFuncSpecial(-3)
        
        # evev/odd +-*/ wrong_grid
        self.assertTrue((test_even_a + wrong_grid).is_special())
        self.assertTrue((test_odd_a + wrong_grid).is_special())
        self.assertTrue((test_even_a - wrong_grid).is_special())
        self.assertTrue((test_odd_a - wrong_grid).is_special())
        self.assertTrue((test_even_a * wrong_grid).is_special())
        self.assertTrue((test_odd_a * wrong_grid).is_special())
        self.assertTrue((test_even_a / wrong_grid).is_special())
        self.assertTrue((test_odd_a / wrong_grid).is_special())
        # evev/odd +-*/ wrong_nfp
        self.assertTrue((test_even_a + test_odd_a_wrong_nfp).is_special())
        self.assertTrue((test_odd_a + test_odd_a_wrong_nfp).is_special())
        self.assertTrue((test_even_a - test_odd_a_wrong_nfp).is_special())
        self.assertTrue((test_odd_a - test_odd_a_wrong_nfp).is_special())
        self.assertTrue((test_even_a * test_odd_a_wrong_nfp).is_special())
        self.assertTrue((test_odd_a * test_odd_a_wrong_nfp).is_special())
        self.assertTrue((test_even_a / test_odd_a_wrong_nfp).is_special())
        self.assertTrue((test_odd_a / test_odd_a_wrong_nfp).is_special())
        # evev/odd +-* single_col
        self.assertTrue(not (test_even_a + single_col).is_special())
        self.assertTrue((test_odd_a + single_col).is_special())
        self.assertTrue(not (test_even_a - single_col).is_special())
        self.assertTrue((test_odd_a - single_col).is_special())
        self.assertTrue(not (test_even_a * single_col).is_special())
        self.assertTrue(not (test_odd_a * single_col).is_special())
        # evev/odd +-*/ wrong_nfp
        self.assertTrue((test_even_a + test_odd_a_wrong_nfp).is_special())
        self.assertTrue((test_odd_a + test_odd_a_wrong_nfp).is_special())
        self.assertTrue((test_even_a - test_odd_a_wrong_nfp).is_special())
        self.assertTrue((test_odd_a - test_odd_a_wrong_nfp).is_special())
        self.assertTrue((test_even_a * test_odd_a_wrong_nfp).is_special())
        self.assertTrue((test_odd_a * test_odd_a_wrong_nfp).is_special())
        self.assertTrue((test_even_a / test_odd_a_wrong_nfp).is_special())
        self.assertTrue((test_odd_a / test_odd_a_wrong_nfp).is_special())
        self.assertTrue(not (test_even_a+test_even_b).is_special())
        self.assertTrue(not (test_odd_a+test_odd_b).is_special())
        self.assertTrue(not (test_even_a+1).is_special())
        self.assertTrue(not (test_even_a+single_row).is_special())
        self.assertTrue((test_odd_a+test_even_b).nfp==-4)
        self.assertTrue(jnp.all((test_odd_a+zero).content==test_odd_a.content))
        self.assertTrue(jnp.all((test_even_a+zero).content==test_even_a.content))
        self.assertTrue((test_odd_a+single_row).nfp==-4)
        self.assertTrue(jnp.isnan(ChiPhiFunc(a, -1).content))
        self.assertTrue((6+zero)==6)
        self.assertTrue((zero+zero).nfp==0)
        self.assertTrue((zero+null_a).nfp==-2)
        self.assertTrue((null_a+zero).nfp==-2)
        self.assertTrue((null_a+null_b).nfp==-203)
        self.assertTrue((test_even_a*test_even_b).content.shape[0]==7)
        self.assertTrue((test_even_a*2).content.shape[0]==5)
        self.assertTrue((test_even_a*zero).nfp==0)
        self.assertTrue((test_even_a*zero).nfp==0)
        self.assertTrue((zero*test_even_a).nfp==0)
        self.assertTrue((zero*zero).nfp==0)
        self.assertTrue((zero*null_a).nfp==-2)
        self.assertTrue((null_a*zero).nfp==-2)
        self.assertTrue((null_a*null_b).nfp==-203)
        self.assertTrue((test_even_a/test_even_b).is_special())
        self.assertTrue((test_even_a/2).content.shape[0]==5)
        self.assertTrue((2/test_even_b).is_special())
        self.assertTrue(not (test_even_a/single_row).is_special())
        self.assertTrue(not (2/single_row).is_special())
        self.assertTrue((2/zero).nfp==-8)
        self.assertTrue((zero/2).nfp==0)
        self.assertTrue((test_even_a/zero).nfp==-8)
        self.assertTrue((zero/test_even_a).nfp==0)
        self.assertTrue((null_a/null_b).nfp==-203)
        # self.assertEqual(divide_by_three(12), 4)

    def test_arithmetics(self):
        '''
        Testing arithmetics such as +, -, *, /, ...
        '''
        # Defining test vars
        # Generating 2 random test cases and answers
        nfp = np.random.randint(4)+2

        # Creating 2 random ChiPhiFunc's for testing
        rands1 = np.random.randint(5, size=12)
        func1 = np.vectorize(lambda chi, phi : \
            (rands1[0]/10*np.sin(rands1[1]*nfp*phi) + rands1[2]/10*np.cos(rands1[3]*nfp*phi))*np.sin(2*chi)+\
            rands1[4]/10*np.sin(rands1[5]*nfp*phi) + rands1[6]/10*np.cos(rands1[7]*nfp*phi)+\
            (rands1[8]/10*np.sin(rands1[9]*nfp*phi) + rands1[10]/10*np.cos(rands1[11]*nfp*phi))*np.cos(2*chi))

        content1 = np.array([
            rands1[0]/10*np.sin(rands1[1]*points) + rands1[2]/10*np.cos(rands1[3]*points),
            rands1[4]/10*np.sin(rands1[5]*points) + rands1[6]/10*np.cos(rands1[7]*points),
            rands1[8]/10*np.sin(rands1[9]*points) + rands1[10]/10*np.cos(rands1[11]*points)
        ])


        rands2 = np.random.randint(5, size=12)
        func2 = np.vectorize(lambda chi, phi : \
            (rands2[0]/10*np.sin(rands2[1]*nfp*phi) + rands2[2]/10*np.cos(rands2[3]*nfp*phi))*np.sin(2*chi)+\
            rands2[4]/10*np.sin(rands2[5]*nfp*phi) + rands2[6]/10*np.cos(rands2[7]*nfp*phi)+\
            (rands2[8]/10*np.sin(rands2[9]*nfp*phi) + rands2[10]/10*np.cos(rands2[11]*nfp*phi))*np.cos(2*chi))
        content2 = np.array([
            rands2[0]/10*np.sin(rands2[1]*points) + rands2[2]/10*np.cos(rands2[3]*points),
            rands2[4]/10*np.sin(rands2[5]*points) + rands2[6]/10*np.cos(rands2[7]*points),
            rands2[8]/10*np.sin(rands2[9]*points) + rands2[10]/10*np.cos(rands2[11]*points)
        ])

        randsodd = np.random.randint(5, size=16)
        funcodd = np.vectorize(lambda chi, phi : \
            (randsodd[0]/10*np.sin(randsodd[1]*nfp*phi) + randsodd[2]/10*np.cos(randsodd[3]*nfp*phi))*np.sin(3*chi)+\
            (randsodd[4]/10*np.sin(randsodd[5]*nfp*phi) + randsodd[6]/10*np.cos(randsodd[7]*nfp*phi))*np.sin(1*chi)+\
            (randsodd[8]/10*np.sin(randsodd[9]*nfp*phi) + randsodd[10]/10*np.cos(randsodd[11]*nfp*phi))*np.cos(1*chi)+\
            (randsodd[12]/10*np.sin(randsodd[13]*nfp*phi) + randsodd[14]/10*np.cos(randsodd[15]*nfp*phi))*np.cos(3*chi))
        contentodd = np.array([
            randsodd[0]/10*np.sin(randsodd[1]*points) + randsodd[2]/10*np.cos(randsodd[3]*points),
            randsodd[4]/10*np.sin(randsodd[5]*points) + randsodd[6]/10*np.cos(randsodd[7]*points),
            randsodd[8]/10*np.sin(randsodd[9]*points) + randsodd[10]/10*np.cos(randsodd[11]*points),
            randsodd[12]/10*np.sin(randsodd[13]*points) + randsodd[14]/10*np.cos(randsodd[15]*points)
        ])


        rands3 = np.random.randint(low=1, high=5, size=2)
        func_no_chi = np.vectorize(lambda chi, phi : 
            np.sin(rands3[0]*nfp*phi) + np.cos(rands3[1]*nfp*phi)+2)
        content_no_chi = np.array([
            np.sin(rands3[0]*points) + np.cos(rands3[1]*points) + 2
        ])
        test1 = ChiPhiFunc(content1, nfp, trig_mode = True)
        test2 = ChiPhiFunc(content2, nfp, trig_mode = True)
        testodd = ChiPhiFunc(contentodd, nfp, trig_mode = True)
        test_no_chi = ChiPhiFunc(content_no_chi, nfp, trig_mode = True)

        funcodd_result = evaluate(funcodd)
        func1_result = evaluate(func1)
        func2_result = evaluate(func2)
        func_no_chi_result = evaluate(func_no_chi)

        # Testing lambda again
        self.assertTrue(jnp.all(jnp.isclose(evaluate_ChiPhiFunc(testodd), (funcodd_result))))
        # Testing +:
        # Odd-odd:
        self.assertTrue(jnp.all(jnp.isclose(evaluate_ChiPhiFunc(testodd+testodd), (funcodd_result+funcodd_result))))
        # Even-even:
        self.assertTrue(jnp.all(jnp.isclose(evaluate_ChiPhiFunc(test1+test2+1), (func1_result+func2_result+1))))
        # Testing -:
        self.assertTrue(jnp.all(jnp.isclose(evaluate_ChiPhiFunc(test1-test2-test_no_chi), (func1_result-func2_result-func_no_chi_result))))
        # Testing *:
        # Odd-odd:
        self.assertTrue(jnp.all(jnp.isclose(evaluate_ChiPhiFunc(testodd*testodd), (funcodd_result*funcodd_result))))
        # Even-even:
        self.assertTrue(jnp.all(jnp.isclose(evaluate_ChiPhiFunc(test1*test2), (func1_result*func2_result))))
        # Even-odd:
        self.assertTrue(jnp.all(jnp.isclose(evaluate_ChiPhiFunc(test1*testodd), (func1_result*funcodd_result))))
        # Testing /:
        # Even:
        self.assertTrue(jnp.all(jnp.isclose(evaluate_ChiPhiFunc(test1/(test_no_chi+10)), (func1_result/(func_no_chi_result+10)))))
        # Odd:
        self.assertTrue(jnp.all(jnp.isclose(evaluate_ChiPhiFunc(testodd/(test_no_chi+10)), (funcodd_result/(func_no_chi_result+10)))))
        # Testing **:
        self.assertTrue(jnp.all(jnp.isclose(evaluate_ChiPhiFunc(test1**3), (func1_result*func1_result*func1_result))))
        self.assertTrue(jnp.all(jnp.isclose(evaluate_ChiPhiFunc(testodd**3), (funcodd_result*funcodd_result*funcodd_result))))

        print('Testing +:')
        print('Odd-odd:')
        print_fractional_error(evaluate_ChiPhiFunc(testodd+testodd), (funcodd_result+funcodd_result))
        print('Even-even:')
        print_fractional_error(evaluate_ChiPhiFunc(test1+test2+1), (func1_result+func2_result+1))

        print('Testing -:')
        print_fractional_error(evaluate_ChiPhiFunc(test1-test2-test_no_chi), (func1_result-func2_result-func_no_chi_result))

        print('Testing *:')
        print('Odd-odd:')
        print_fractional_error(evaluate_ChiPhiFunc(testodd*testodd), (funcodd_result*funcodd_result))
        print('Even-even:')
        print_fractional_error(evaluate_ChiPhiFunc(test1*test2), (func1_result*func2_result))
        print('Even-odd:')
        print_fractional_error(evaluate_ChiPhiFunc(test1*testodd), (func1_result*funcodd_result))

        print('Testing /:')
        print('Even:')
        print_fractional_error(evaluate_ChiPhiFunc(test1/(test_no_chi+10)), (func1_result/(func_no_chi_result+10)))
        print('Odd:')
        print_fractional_error(evaluate_ChiPhiFunc(testodd/(test_no_chi+10)), (funcodd_result/(func_no_chi_result+10)))


        print('Testing **:')
        print_fractional_error(evaluate_ChiPhiFunc(test1**3), (func1_result*func1_result*func1_result))
        print_fractional_error(evaluate_ChiPhiFunc(testodd**3), (funcodd_result*funcodd_result*funcodd_result))
        
if __name__ == '__main__':
    unittest.main()