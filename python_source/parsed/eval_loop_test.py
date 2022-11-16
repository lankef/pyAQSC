# Evaluating loop. 
# Uses Xn-1, Yn-1, Zn-1,  B_theta_n, B_psi_n-3, iota_coef (n-1)/2 or (n-2)/2
# Must be evaluated with Z_coef_cp[n] = 0, p_perp_coef_cp[n] = 0
# B_psi_coef_cp[n-2] = 0 and B_theta_coef_cp[n] = 0 
from math import floor, ceil
from math_utilities import *
import chiphifunc
def eval_loop_test(n, X_coef_cp, Y_coef_cp, Z_coef_cp, \
    B_theta_coef_cp, B_psi_coef_cp, B_alpha_coef, B_denom_coef_c, \
    p_perp_coef_cp, Delta_coef_cp, kap_p, dl_p, tau_p, iota_coef):    
    def sum_arg_323(i292):
        # Child args for sum_arg_323
        return(X_coef_cp[i292]*diff(Y_coef_cp[n-i292],'chi',1))
    
    def sum_arg_322(i288):
        # Child args for sum_arg_322
        return(Y_coef_cp[i288]*diff(X_coef_cp[n-i288],'chi',1))
    
    def sum_arg_321(i292):
        # Child args for sum_arg_321
        return(X_coef_cp[i292]*diff(Y_coef_cp[n-i292],'chi',1,'phi',1)+diff(X_coef_cp[i292],'phi',1)*diff(Y_coef_cp[n-i292],'chi',1))
    
    def sum_arg_320(i292):
        # Child args for sum_arg_320
        return(X_coef_cp[i292]*diff(Y_coef_cp[n-i292],'chi',1))
    
    def sum_arg_319(i288):
        # Child args for sum_arg_319
        return(Y_coef_cp[i288]*diff(X_coef_cp[n-i288],'chi',1,'phi',1)+diff(Y_coef_cp[i288],'phi',1)*diff(X_coef_cp[n-i288],'chi',1))
    
    def sum_arg_318(i288):
        # Child args for sum_arg_318
        return(Y_coef_cp[i288]*diff(X_coef_cp[n-i288],'chi',1))
    
    def sum_arg_317(i294):
        # Child args for sum_arg_317
        return(X_coef_cp[i294]*diff(Z_coef_cp[n-i294],'chi',1,'phi',1)+diff(X_coef_cp[i294],'phi',1)*diff(Z_coef_cp[n-i294],'chi',1))
    
    def sum_arg_316(i294):
        # Child args for sum_arg_316
        return(X_coef_cp[i294]*diff(Z_coef_cp[n-i294],'chi',1))
    
    def sum_arg_315(i294):
        # Child args for sum_arg_315
        return(X_coef_cp[i294]*diff(Z_coef_cp[n-i294],'chi',1))
    
    def sum_arg_314(i290):
        # Child args for sum_arg_314
        return(Z_coef_cp[i290]*diff(X_coef_cp[n-i290],'chi',1,'phi',1)+diff(Z_coef_cp[i290],'phi',1)*diff(X_coef_cp[n-i290],'chi',1))
    
    def sum_arg_313(i290):
        # Child args for sum_arg_313
        return(Z_coef_cp[i290]*diff(X_coef_cp[n-i290],'chi',1))
    
    def sum_arg_312(i290):
        # Child args for sum_arg_312
        return(Z_coef_cp[i290]*diff(X_coef_cp[n-i290],'chi',1))
    
    def sum_arg_311(i285):
        # Child args for sum_arg_311    
        def sum_arg_310(i286):
            # Child args for sum_arg_310
            return(diff(Z_coef_cp[i286],'chi',1)*diff(Z_coef_cp[(-n)-i286+2*i285],'chi',1,'phi',1)*is_seq(n-i285,i285-i286)+diff(Z_coef_cp[i286],'chi',1,'phi',1)*diff(Z_coef_cp[(-n)-i286+2*i285],'chi',1)*is_seq(n-i285,i285-i286))
        
        return(is_seq(0,n-i285)*iota_coef[n-i285]*is_integer(n-i285)*py_sum(sum_arg_310,0,i285))
    
    def sum_arg_309(i284):
        # Child args for sum_arg_309
        return(diff(Z_coef_cp[i284],'chi',1)*diff(Z_coef_cp[n-i284],'phi',2)+diff(Z_coef_cp[i284],'chi',1,'phi',1)*diff(Z_coef_cp[n-i284],'phi',1))
    
    def sum_arg_308(i281):
        # Child args for sum_arg_308    
        def sum_arg_307(i282):
            # Child args for sum_arg_307
            return(diff(Y_coef_cp[i282],'chi',1)*diff(Y_coef_cp[(-n)-i282+2*i281],'chi',1,'phi',1)*is_seq(n-i281,i281-i282)+diff(Y_coef_cp[i282],'chi',1,'phi',1)*diff(Y_coef_cp[(-n)-i282+2*i281],'chi',1)*is_seq(n-i281,i281-i282))
        
        return(is_seq(0,n-i281)*iota_coef[n-i281]*is_integer(n-i281)*py_sum(sum_arg_307,0,i281))
    
    def sum_arg_306(i280):
        # Child args for sum_arg_306
        return(diff(Y_coef_cp[i280],'chi',1)*diff(Y_coef_cp[n-i280],'phi',2)+diff(Y_coef_cp[i280],'chi',1,'phi',1)*diff(Y_coef_cp[n-i280],'phi',1))
    
    def sum_arg_305(i277):
        # Child args for sum_arg_305    
        def sum_arg_304(i278):
            # Child args for sum_arg_304
            return(diff(X_coef_cp[i278],'chi',1)*diff(X_coef_cp[(-n)-i278+2*i277],'chi',1,'phi',1)*is_seq(n-i277,i277-i278)+diff(X_coef_cp[i278],'chi',1,'phi',1)*diff(X_coef_cp[(-n)-i278+2*i277],'chi',1)*is_seq(n-i277,i277-i278))
        
        return(is_seq(0,n-i277)*iota_coef[n-i277]*is_integer(n-i277)*py_sum(sum_arg_304,0,i277))
    
    def sum_arg_303(i276):
        # Child args for sum_arg_303
        return(diff(X_coef_cp[i276],'chi',1)*diff(X_coef_cp[n-i276],'phi',2)+diff(X_coef_cp[i276],'chi',1,'phi',1)*diff(X_coef_cp[n-i276],'phi',1))
    
    def sum_arg_302(i209):
        # Child args for sum_arg_302    
        def sum_arg_301(i210):
            # Child args for sum_arg_301
            return(diff(B_theta_coef_cp[i210],'phi',1)*B_denom_coef_c[(-n)-i210+2*i209]*is_seq(n-i209,i209-i210))
        
        return(is_seq(0,n-i209)*B_alpha_coef[n-i209]*is_integer(n-i209)*py_sum(sum_arg_301,0,i209))
    
    def sum_arg_300(i260):
        # Child args for sum_arg_300
        return(i260*X_coef_cp[i260]*diff(Y_coef_cp[n-i260],'chi',1)+i260*diff(X_coef_cp[i260],'chi',1)*Y_coef_cp[n-i260])
    
    def sum_arg_299(i258):
        # Child args for sum_arg_299
        return(X_coef_cp[i258]*(n-i258)*diff(Y_coef_cp[n-i258],'chi',1)+diff(X_coef_cp[i258],'chi',1)*(n-i258)*Y_coef_cp[n-i258])
    
    def sum_arg_298(i260):
        # Child args for sum_arg_298
        return(i260*diff(X_coef_cp[i260],'chi',1)*diff(Y_coef_cp[n-i260],'phi',1)+i260*X_coef_cp[i260]*diff(Y_coef_cp[n-i260],'chi',1,'phi',1)+i260*diff(X_coef_cp[i260],'phi',1)*diff(Y_coef_cp[n-i260],'chi',1)+i260*diff(X_coef_cp[i260],'chi',1,'phi',1)*Y_coef_cp[n-i260])
    
    def sum_arg_297(i260):
        # Child args for sum_arg_297
        return(i260*X_coef_cp[i260]*diff(Y_coef_cp[n-i260],'chi',1)+i260*diff(X_coef_cp[i260],'chi',1)*Y_coef_cp[n-i260])
    
    def sum_arg_296(i258):
        # Child args for sum_arg_296
        return(diff(X_coef_cp[i258],'chi',1)*(n-i258)*diff(Y_coef_cp[n-i258],'phi',1)+X_coef_cp[i258]*(n-i258)*diff(Y_coef_cp[n-i258],'chi',1,'phi',1)+diff(X_coef_cp[i258],'phi',1)*(n-i258)*diff(Y_coef_cp[n-i258],'chi',1)+diff(X_coef_cp[i258],'chi',1,'phi',1)*(n-i258)*Y_coef_cp[n-i258])
    
    def sum_arg_295(i258):
        # Child args for sum_arg_295
        return(X_coef_cp[i258]*(n-i258)*diff(Y_coef_cp[n-i258],'chi',1)+diff(X_coef_cp[i258],'chi',1)*(n-i258)*Y_coef_cp[n-i258])
    
    def sum_arg_294(i273):
        # Child args for sum_arg_294    
        def sum_arg_293(i274):
            # Child args for sum_arg_293
            return(i274*Z_coef_cp[i274]*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',2,'phi',1)*is_seq(n-i273,i273-i274)+i274*diff(Z_coef_cp[i274],'phi',1)*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',2)*is_seq(n-i273,i273-i274)+i274*diff(Z_coef_cp[i274],'chi',1)*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',1,'phi',1)*is_seq(n-i273,i273-i274)+i274*diff(Z_coef_cp[i274],'chi',1,'phi',1)*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',1)*is_seq(n-i273,i273-i274))
        
        return(is_seq(0,n-i273)*iota_coef[n-i273]*is_integer(n-i273)*py_sum(sum_arg_293,0,i273))
    
    def sum_arg_292(i271):
        # Child args for sum_arg_292    
        def sum_arg_291(i272):
            # Child args for sum_arg_291
            return(i272*Y_coef_cp[i272]*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',2,'phi',1)*is_seq(n-i271,i271-i272)+i272*diff(Y_coef_cp[i272],'phi',1)*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',2)*is_seq(n-i271,i271-i272)+i272*diff(Y_coef_cp[i272],'chi',1)*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',1,'phi',1)*is_seq(n-i271,i271-i272)+i272*diff(Y_coef_cp[i272],'chi',1,'phi',1)*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',1)*is_seq(n-i271,i271-i272))
        
        return(is_seq(0,n-i271)*iota_coef[n-i271]*is_integer(n-i271)*py_sum(sum_arg_291,0,i271))
    
    def sum_arg_290(i269):
        # Child args for sum_arg_290    
        def sum_arg_289(i270):
            # Child args for sum_arg_289
            return(i270*X_coef_cp[i270]*diff(X_coef_cp[(-n)-i270+2*i269],'chi',2,'phi',1)*is_seq(n-i269,i269-i270)+i270*diff(X_coef_cp[i270],'phi',1)*diff(X_coef_cp[(-n)-i270+2*i269],'chi',2)*is_seq(n-i269,i269-i270)+i270*diff(X_coef_cp[i270],'chi',1)*diff(X_coef_cp[(-n)-i270+2*i269],'chi',1,'phi',1)*is_seq(n-i269,i269-i270)+i270*diff(X_coef_cp[i270],'chi',1,'phi',1)*diff(X_coef_cp[(-n)-i270+2*i269],'chi',1)*is_seq(n-i269,i269-i270))
        
        return(is_seq(0,n-i269)*iota_coef[n-i269]*is_integer(n-i269)*py_sum(sum_arg_289,0,i269))
    
    def sum_arg_288(i268):
        # Child args for sum_arg_288
        return(i268*diff(Z_coef_cp[i268],'chi',1)*diff(Z_coef_cp[n-i268],'phi',2)+i268*diff(Z_coef_cp[i268],'chi',1,'phi',1)*diff(Z_coef_cp[n-i268],'phi',1)+i268*Z_coef_cp[i268]*diff(Z_coef_cp[n-i268],'chi',1,'phi',2)+i268*diff(Z_coef_cp[i268],'phi',1)*diff(Z_coef_cp[n-i268],'chi',1,'phi',1))
    
    def sum_arg_287(i266):
        # Child args for sum_arg_287
        return(i266*diff(Y_coef_cp[i266],'chi',1)*diff(Y_coef_cp[n-i266],'phi',2)+i266*diff(Y_coef_cp[i266],'chi',1,'phi',1)*diff(Y_coef_cp[n-i266],'phi',1)+i266*Y_coef_cp[i266]*diff(Y_coef_cp[n-i266],'chi',1,'phi',2)+i266*diff(Y_coef_cp[i266],'phi',1)*diff(Y_coef_cp[n-i266],'chi',1,'phi',1))
    
    def sum_arg_286(i264):
        # Child args for sum_arg_286
        return(i264*diff(X_coef_cp[i264],'chi',1)*diff(X_coef_cp[n-i264],'phi',2)+i264*diff(X_coef_cp[i264],'chi',1,'phi',1)*diff(X_coef_cp[n-i264],'phi',1)+i264*X_coef_cp[i264]*diff(X_coef_cp[n-i264],'chi',1,'phi',2)+i264*diff(X_coef_cp[i264],'phi',1)*diff(X_coef_cp[n-i264],'chi',1,'phi',1))
    
    def sum_arg_285(i262):
        # Child args for sum_arg_285
        return(i262*diff(X_coef_cp[i262],'chi',1)*diff(Z_coef_cp[n-i262],'phi',1)+i262*X_coef_cp[i262]*diff(Z_coef_cp[n-i262],'chi',1,'phi',1)+i262*diff(X_coef_cp[i262],'phi',1)*diff(Z_coef_cp[n-i262],'chi',1)+i262*diff(X_coef_cp[i262],'chi',1,'phi',1)*Z_coef_cp[n-i262])
    
    def sum_arg_284(i262):
        # Child args for sum_arg_284
        return(i262*X_coef_cp[i262]*diff(Z_coef_cp[n-i262],'chi',1)+i262*diff(X_coef_cp[i262],'chi',1)*Z_coef_cp[n-i262])
    
    def sum_arg_283(i262):
        # Child args for sum_arg_283
        return(i262*X_coef_cp[i262]*diff(Z_coef_cp[n-i262],'chi',1)+i262*diff(X_coef_cp[i262],'chi',1)*Z_coef_cp[n-i262])
    
    def sum_arg_282(i256):
        # Child args for sum_arg_282
        return(diff(X_coef_cp[i256],'chi',1)*(n-i256)*diff(Z_coef_cp[n-i256],'phi',1)+X_coef_cp[i256]*(n-i256)*diff(Z_coef_cp[n-i256],'chi',1,'phi',1)+diff(X_coef_cp[i256],'phi',1)*(n-i256)*diff(Z_coef_cp[n-i256],'chi',1)+diff(X_coef_cp[i256],'chi',1,'phi',1)*(n-i256)*Z_coef_cp[n-i256])
    
    def sum_arg_281(i256):
        # Child args for sum_arg_281
        return(X_coef_cp[i256]*(n-i256)*diff(Z_coef_cp[n-i256],'chi',1)+diff(X_coef_cp[i256],'chi',1)*(n-i256)*Z_coef_cp[n-i256])
    
    def sum_arg_280(i256):
        # Child args for sum_arg_280
        return(X_coef_cp[i256]*(n-i256)*diff(Z_coef_cp[n-i256],'chi',1)+diff(X_coef_cp[i256],'chi',1)*(n-i256)*Z_coef_cp[n-i256])
    
    def sum_arg_279(i201):
        # Child args for sum_arg_279    
        def sum_arg_278(i202):
            # Child args for sum_arg_278
            return(diff(B_psi_coef_cp[i202],'chi',1,'phi',1)*B_denom_coef_c[(-n)-i202+2*i201+2]*is_seq(n-i201-2,i201-i202)+diff(B_psi_coef_cp[i202],'phi',1)*diff(B_denom_coef_c[(-n)-i202+2*i201+2],'chi',1)*is_seq(n-i201-2,i201-i202))
        
        return(is_seq(0,n-i201-2)*B_alpha_coef[n-i201-2]*is_integer(n-i201-2)*py_sum(sum_arg_278,0,i201))
    
    def sum_arg_277(i292):
        # Child args for sum_arg_277
        return(X_coef_cp[i292]*diff(Y_coef_cp[n-i292],'chi',1))
    
    def sum_arg_276(i288):
        # Child args for sum_arg_276
        return(Y_coef_cp[i288]*diff(X_coef_cp[n-i288],'chi',1))
    
    def sum_arg_275(i294):
        # Child args for sum_arg_275
        return(X_coef_cp[i294]*diff(Z_coef_cp[n-i294],'chi',1))
    
    def sum_arg_274(i290):
        # Child args for sum_arg_274
        return(Z_coef_cp[i290]*diff(X_coef_cp[n-i290],'chi',1))
    
    def sum_arg_273(i285):
        # Child args for sum_arg_273    
        def sum_arg_272(i286):
            # Child args for sum_arg_272
            return(diff(Z_coef_cp[i286],'chi',1)*diff(Z_coef_cp[(-n)-i286+2*i285],'chi',1)*is_seq(n-i285,i285-i286))
        
        return(is_seq(0,n-i285)*iota_coef[n-i285]*is_integer(n-i285)*py_sum(sum_arg_272,0,i285))
    
    def sum_arg_271(i284):
        # Child args for sum_arg_271
        return(diff(Z_coef_cp[i284],'chi',1)*diff(Z_coef_cp[n-i284],'phi',1))
    
    def sum_arg_270(i281):
        # Child args for sum_arg_270    
        def sum_arg_269(i282):
            # Child args for sum_arg_269
            return(diff(Y_coef_cp[i282],'chi',1)*diff(Y_coef_cp[(-n)-i282+2*i281],'chi',1)*is_seq(n-i281,i281-i282))
        
        return(is_seq(0,n-i281)*iota_coef[n-i281]*is_integer(n-i281)*py_sum(sum_arg_269,0,i281))
    
    def sum_arg_268(i280):
        # Child args for sum_arg_268
        return(diff(Y_coef_cp[i280],'chi',1)*diff(Y_coef_cp[n-i280],'phi',1))
    
    def sum_arg_267(i277):
        # Child args for sum_arg_267    
        def sum_arg_266(i278):
            # Child args for sum_arg_266
            return(diff(X_coef_cp[i278],'chi',1)*diff(X_coef_cp[(-n)-i278+2*i277],'chi',1)*is_seq(n-i277,i277-i278))
        
        return(is_seq(0,n-i277)*iota_coef[n-i277]*is_integer(n-i277)*py_sum(sum_arg_266,0,i277))
    
    def sum_arg_265(i276):
        # Child args for sum_arg_265
        return(diff(X_coef_cp[i276],'chi',1)*diff(X_coef_cp[n-i276],'phi',1))
    
    def sum_arg_264(i209):
        # Child args for sum_arg_264    
        def sum_arg_263(i210):
            # Child args for sum_arg_263
            return(B_theta_coef_cp[i210]*B_denom_coef_c[(-n)-i210+2*i209]*is_seq(n-i209,i209-i210))
        
        return(is_seq(0,n-i209)*B_alpha_coef[n-i209]*is_integer(n-i209)*py_sum(sum_arg_263,0,i209))
    
    def sum_arg_262(i260):
        # Child args for sum_arg_262
        return(i260*X_coef_cp[i260]*diff(Y_coef_cp[n-i260],'chi',1)+i260*diff(X_coef_cp[i260],'chi',1)*Y_coef_cp[n-i260])
    
    def sum_arg_261(i258):
        # Child args for sum_arg_261
        return(X_coef_cp[i258]*(n-i258)*diff(Y_coef_cp[n-i258],'chi',1)+diff(X_coef_cp[i258],'chi',1)*(n-i258)*Y_coef_cp[n-i258])
    
    def sum_arg_260(i273):
        # Child args for sum_arg_260    
        def sum_arg_259(i274):
            # Child args for sum_arg_259
            return(i274*Z_coef_cp[i274]*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',2)*is_seq(n-i273,i273-i274)+i274*diff(Z_coef_cp[i274],'chi',1)*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',1)*is_seq(n-i273,i273-i274))
        
        return(is_seq(0,n-i273)*iota_coef[n-i273]*is_integer(n-i273)*py_sum(sum_arg_259,0,i273))
    
    def sum_arg_258(i271):
        # Child args for sum_arg_258    
        def sum_arg_257(i272):
            # Child args for sum_arg_257
            return(i272*Y_coef_cp[i272]*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',2)*is_seq(n-i271,i271-i272)+i272*diff(Y_coef_cp[i272],'chi',1)*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',1)*is_seq(n-i271,i271-i272))
        
        return(is_seq(0,n-i271)*iota_coef[n-i271]*is_integer(n-i271)*py_sum(sum_arg_257,0,i271))
    
    def sum_arg_256(i269):
        # Child args for sum_arg_256    
        def sum_arg_255(i270):
            # Child args for sum_arg_255
            return(i270*X_coef_cp[i270]*diff(X_coef_cp[(-n)-i270+2*i269],'chi',2)*is_seq(n-i269,i269-i270)+i270*diff(X_coef_cp[i270],'chi',1)*diff(X_coef_cp[(-n)-i270+2*i269],'chi',1)*is_seq(n-i269,i269-i270))
        
        return(is_seq(0,n-i269)*iota_coef[n-i269]*is_integer(n-i269)*py_sum(sum_arg_255,0,i269))
    
    def sum_arg_254(i268):
        # Child args for sum_arg_254
        return(i268*diff(Z_coef_cp[i268],'chi',1)*diff(Z_coef_cp[n-i268],'phi',1)+i268*Z_coef_cp[i268]*diff(Z_coef_cp[n-i268],'chi',1,'phi',1))
    
    def sum_arg_253(i266):
        # Child args for sum_arg_253
        return(i266*diff(Y_coef_cp[i266],'chi',1)*diff(Y_coef_cp[n-i266],'phi',1)+i266*Y_coef_cp[i266]*diff(Y_coef_cp[n-i266],'chi',1,'phi',1))
    
    def sum_arg_252(i264):
        # Child args for sum_arg_252
        return(i264*diff(X_coef_cp[i264],'chi',1)*diff(X_coef_cp[n-i264],'phi',1)+i264*X_coef_cp[i264]*diff(X_coef_cp[n-i264],'chi',1,'phi',1))
    
    def sum_arg_251(i262):
        # Child args for sum_arg_251
        return(i262*X_coef_cp[i262]*diff(Z_coef_cp[n-i262],'chi',1)+i262*diff(X_coef_cp[i262],'chi',1)*Z_coef_cp[n-i262])
    
    def sum_arg_250(i256):
        # Child args for sum_arg_250
        return(X_coef_cp[i256]*(n-i256)*diff(Z_coef_cp[n-i256],'chi',1)+diff(X_coef_cp[i256],'chi',1)*(n-i256)*Z_coef_cp[n-i256])
    
    def sum_arg_249(i201):
        # Child args for sum_arg_249    
        def sum_arg_248(i202):
            # Child args for sum_arg_248
            return(diff(B_psi_coef_cp[i202],'chi',1)*B_denom_coef_c[(-n)-i202+2*i201+2]*is_seq(n-i201-2,i201-i202)+B_psi_coef_cp[i202]*diff(B_denom_coef_c[(-n)-i202+2*i201+2],'chi',1)*is_seq(n-i201-2,i201-i202))
        
        return(is_seq(0,n-i201-2)*B_alpha_coef[n-i201-2]*is_integer(n-i201-2)*py_sum(sum_arg_248,0,i201))
    
    def sum_arg_247(i292):
        # Child args for sum_arg_247
        return(X_coef_cp[i292]*diff(Y_coef_cp[n-i292],'chi',1))
    
    def sum_arg_246(i288):
        # Child args for sum_arg_246
        return(Y_coef_cp[i288]*diff(X_coef_cp[n-i288],'chi',1))
    
    def sum_arg_245(i292):
        # Child args for sum_arg_245
        return(X_coef_cp[i292]*diff(Y_coef_cp[n-i292],'chi',1,'phi',1)+diff(X_coef_cp[i292],'phi',1)*diff(Y_coef_cp[n-i292],'chi',1))
    
    def sum_arg_244(i292):
        # Child args for sum_arg_244
        return(X_coef_cp[i292]*diff(Y_coef_cp[n-i292],'chi',1))
    
    def sum_arg_243(i288):
        # Child args for sum_arg_243
        return(Y_coef_cp[i288]*diff(X_coef_cp[n-i288],'chi',1,'phi',1)+diff(Y_coef_cp[i288],'phi',1)*diff(X_coef_cp[n-i288],'chi',1))
    
    def sum_arg_242(i288):
        # Child args for sum_arg_242
        return(Y_coef_cp[i288]*diff(X_coef_cp[n-i288],'chi',1))
    
    def sum_arg_241(i294):
        # Child args for sum_arg_241
        return(X_coef_cp[i294]*diff(Z_coef_cp[n-i294],'chi',1,'phi',1)+diff(X_coef_cp[i294],'phi',1)*diff(Z_coef_cp[n-i294],'chi',1))
    
    def sum_arg_240(i294):
        # Child args for sum_arg_240
        return(X_coef_cp[i294]*diff(Z_coef_cp[n-i294],'chi',1))
    
    def sum_arg_239(i294):
        # Child args for sum_arg_239
        return(X_coef_cp[i294]*diff(Z_coef_cp[n-i294],'chi',1))
    
    def sum_arg_238(i290):
        # Child args for sum_arg_238
        return(Z_coef_cp[i290]*diff(X_coef_cp[n-i290],'chi',1,'phi',1)+diff(Z_coef_cp[i290],'phi',1)*diff(X_coef_cp[n-i290],'chi',1))
    
    def sum_arg_237(i290):
        # Child args for sum_arg_237
        return(Z_coef_cp[i290]*diff(X_coef_cp[n-i290],'chi',1))
    
    def sum_arg_236(i290):
        # Child args for sum_arg_236
        return(Z_coef_cp[i290]*diff(X_coef_cp[n-i290],'chi',1))
    
    def sum_arg_235(i285):
        # Child args for sum_arg_235    
        def sum_arg_234(i286):
            # Child args for sum_arg_234
            return(diff(Z_coef_cp[i286],'chi',1)*diff(Z_coef_cp[(-n)-i286+2*i285],'chi',1,'phi',1)*is_seq(n-i285,i285-i286)+diff(Z_coef_cp[i286],'chi',1,'phi',1)*diff(Z_coef_cp[(-n)-i286+2*i285],'chi',1)*is_seq(n-i285,i285-i286))
        
        return(is_seq(0,n-i285)*iota_coef[n-i285]*is_integer(n-i285)*py_sum(sum_arg_234,0,i285))
    
    def sum_arg_233(i284):
        # Child args for sum_arg_233
        return(diff(Z_coef_cp[i284],'chi',1)*diff(Z_coef_cp[n-i284],'phi',2)+diff(Z_coef_cp[i284],'chi',1,'phi',1)*diff(Z_coef_cp[n-i284],'phi',1))
    
    def sum_arg_232(i281):
        # Child args for sum_arg_232    
        def sum_arg_231(i282):
            # Child args for sum_arg_231
            return(diff(Y_coef_cp[i282],'chi',1)*diff(Y_coef_cp[(-n)-i282+2*i281],'chi',1,'phi',1)*is_seq(n-i281,i281-i282)+diff(Y_coef_cp[i282],'chi',1,'phi',1)*diff(Y_coef_cp[(-n)-i282+2*i281],'chi',1)*is_seq(n-i281,i281-i282))
        
        return(is_seq(0,n-i281)*iota_coef[n-i281]*is_integer(n-i281)*py_sum(sum_arg_231,0,i281))
    
    def sum_arg_230(i280):
        # Child args for sum_arg_230
        return(diff(Y_coef_cp[i280],'chi',1)*diff(Y_coef_cp[n-i280],'phi',2)+diff(Y_coef_cp[i280],'chi',1,'phi',1)*diff(Y_coef_cp[n-i280],'phi',1))
    
    def sum_arg_229(i277):
        # Child args for sum_arg_229    
        def sum_arg_228(i278):
            # Child args for sum_arg_228
            return(diff(X_coef_cp[i278],'chi',1)*diff(X_coef_cp[(-n)-i278+2*i277],'chi',1,'phi',1)*is_seq(n-i277,i277-i278)+diff(X_coef_cp[i278],'chi',1,'phi',1)*diff(X_coef_cp[(-n)-i278+2*i277],'chi',1)*is_seq(n-i277,i277-i278))
        
        return(is_seq(0,n-i277)*iota_coef[n-i277]*is_integer(n-i277)*py_sum(sum_arg_228,0,i277))
    
    def sum_arg_227(i276):
        # Child args for sum_arg_227
        return(diff(X_coef_cp[i276],'chi',1)*diff(X_coef_cp[n-i276],'phi',2)+diff(X_coef_cp[i276],'chi',1,'phi',1)*diff(X_coef_cp[n-i276],'phi',1))
    
    def sum_arg_226(i209):
        # Child args for sum_arg_226    
        def sum_arg_225(i210):
            # Child args for sum_arg_225
            return(diff(B_theta_coef_cp[i210],'phi',1)*B_denom_coef_c[(-n)-i210+2*i209]*is_seq(n-i209,i209-i210))
        
        return(is_seq(0,n-i209)*B_alpha_coef[n-i209]*is_integer(n-i209)*py_sum(sum_arg_225,0,i209))
    
    def sum_arg_224(i260):
        # Child args for sum_arg_224
        return(i260*X_coef_cp[i260]*diff(Y_coef_cp[n-i260],'chi',1)+i260*diff(X_coef_cp[i260],'chi',1)*Y_coef_cp[n-i260])
    
    def sum_arg_223(i258):
        # Child args for sum_arg_223
        return(X_coef_cp[i258]*(n-i258)*diff(Y_coef_cp[n-i258],'chi',1)+diff(X_coef_cp[i258],'chi',1)*(n-i258)*Y_coef_cp[n-i258])
    
    def sum_arg_222(i260):
        # Child args for sum_arg_222
        return(i260*diff(X_coef_cp[i260],'chi',1)*diff(Y_coef_cp[n-i260],'phi',1)+i260*X_coef_cp[i260]*diff(Y_coef_cp[n-i260],'chi',1,'phi',1)+i260*diff(X_coef_cp[i260],'phi',1)*diff(Y_coef_cp[n-i260],'chi',1)+i260*diff(X_coef_cp[i260],'chi',1,'phi',1)*Y_coef_cp[n-i260])
    
    def sum_arg_221(i260):
        # Child args for sum_arg_221
        return(i260*X_coef_cp[i260]*diff(Y_coef_cp[n-i260],'chi',1)+i260*diff(X_coef_cp[i260],'chi',1)*Y_coef_cp[n-i260])
    
    def sum_arg_220(i258):
        # Child args for sum_arg_220
        return(diff(X_coef_cp[i258],'chi',1)*(n-i258)*diff(Y_coef_cp[n-i258],'phi',1)+X_coef_cp[i258]*(n-i258)*diff(Y_coef_cp[n-i258],'chi',1,'phi',1)+diff(X_coef_cp[i258],'phi',1)*(n-i258)*diff(Y_coef_cp[n-i258],'chi',1)+diff(X_coef_cp[i258],'chi',1,'phi',1)*(n-i258)*Y_coef_cp[n-i258])
    
    def sum_arg_219(i258):
        # Child args for sum_arg_219
        return(X_coef_cp[i258]*(n-i258)*diff(Y_coef_cp[n-i258],'chi',1)+diff(X_coef_cp[i258],'chi',1)*(n-i258)*Y_coef_cp[n-i258])
    
    def sum_arg_218(i273):
        # Child args for sum_arg_218    
        def sum_arg_217(i274):
            # Child args for sum_arg_217
            return(i274*Z_coef_cp[i274]*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',2,'phi',1)*is_seq(n-i273,i273-i274)+i274*diff(Z_coef_cp[i274],'phi',1)*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',2)*is_seq(n-i273,i273-i274)+i274*diff(Z_coef_cp[i274],'chi',1)*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',1,'phi',1)*is_seq(n-i273,i273-i274)+i274*diff(Z_coef_cp[i274],'chi',1,'phi',1)*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',1)*is_seq(n-i273,i273-i274))
        
        return(is_seq(0,n-i273)*iota_coef[n-i273]*is_integer(n-i273)*py_sum(sum_arg_217,0,i273))
    
    def sum_arg_216(i271):
        # Child args for sum_arg_216    
        def sum_arg_215(i272):
            # Child args for sum_arg_215
            return(i272*Y_coef_cp[i272]*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',2,'phi',1)*is_seq(n-i271,i271-i272)+i272*diff(Y_coef_cp[i272],'phi',1)*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',2)*is_seq(n-i271,i271-i272)+i272*diff(Y_coef_cp[i272],'chi',1)*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',1,'phi',1)*is_seq(n-i271,i271-i272)+i272*diff(Y_coef_cp[i272],'chi',1,'phi',1)*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',1)*is_seq(n-i271,i271-i272))
        
        return(is_seq(0,n-i271)*iota_coef[n-i271]*is_integer(n-i271)*py_sum(sum_arg_215,0,i271))
    
    def sum_arg_214(i269):
        # Child args for sum_arg_214    
        def sum_arg_213(i270):
            # Child args for sum_arg_213
            return(i270*X_coef_cp[i270]*diff(X_coef_cp[(-n)-i270+2*i269],'chi',2,'phi',1)*is_seq(n-i269,i269-i270)+i270*diff(X_coef_cp[i270],'phi',1)*diff(X_coef_cp[(-n)-i270+2*i269],'chi',2)*is_seq(n-i269,i269-i270)+i270*diff(X_coef_cp[i270],'chi',1)*diff(X_coef_cp[(-n)-i270+2*i269],'chi',1,'phi',1)*is_seq(n-i269,i269-i270)+i270*diff(X_coef_cp[i270],'chi',1,'phi',1)*diff(X_coef_cp[(-n)-i270+2*i269],'chi',1)*is_seq(n-i269,i269-i270))
        
        return(is_seq(0,n-i269)*iota_coef[n-i269]*is_integer(n-i269)*py_sum(sum_arg_213,0,i269))
    
    def sum_arg_212(i268):
        # Child args for sum_arg_212
        return(i268*diff(Z_coef_cp[i268],'chi',1)*diff(Z_coef_cp[n-i268],'phi',2)+i268*diff(Z_coef_cp[i268],'chi',1,'phi',1)*diff(Z_coef_cp[n-i268],'phi',1)+i268*Z_coef_cp[i268]*diff(Z_coef_cp[n-i268],'chi',1,'phi',2)+i268*diff(Z_coef_cp[i268],'phi',1)*diff(Z_coef_cp[n-i268],'chi',1,'phi',1))
    
    def sum_arg_211(i266):
        # Child args for sum_arg_211
        return(i266*diff(Y_coef_cp[i266],'chi',1)*diff(Y_coef_cp[n-i266],'phi',2)+i266*diff(Y_coef_cp[i266],'chi',1,'phi',1)*diff(Y_coef_cp[n-i266],'phi',1)+i266*Y_coef_cp[i266]*diff(Y_coef_cp[n-i266],'chi',1,'phi',2)+i266*diff(Y_coef_cp[i266],'phi',1)*diff(Y_coef_cp[n-i266],'chi',1,'phi',1))
    
    def sum_arg_210(i264):
        # Child args for sum_arg_210
        return(i264*diff(X_coef_cp[i264],'chi',1)*diff(X_coef_cp[n-i264],'phi',2)+i264*diff(X_coef_cp[i264],'chi',1,'phi',1)*diff(X_coef_cp[n-i264],'phi',1)+i264*X_coef_cp[i264]*diff(X_coef_cp[n-i264],'chi',1,'phi',2)+i264*diff(X_coef_cp[i264],'phi',1)*diff(X_coef_cp[n-i264],'chi',1,'phi',1))
    
    def sum_arg_209(i262):
        # Child args for sum_arg_209
        return(i262*diff(X_coef_cp[i262],'chi',1)*diff(Z_coef_cp[n-i262],'phi',1)+i262*X_coef_cp[i262]*diff(Z_coef_cp[n-i262],'chi',1,'phi',1)+i262*diff(X_coef_cp[i262],'phi',1)*diff(Z_coef_cp[n-i262],'chi',1)+i262*diff(X_coef_cp[i262],'chi',1,'phi',1)*Z_coef_cp[n-i262])
    
    def sum_arg_208(i262):
        # Child args for sum_arg_208
        return(i262*X_coef_cp[i262]*diff(Z_coef_cp[n-i262],'chi',1)+i262*diff(X_coef_cp[i262],'chi',1)*Z_coef_cp[n-i262])
    
    def sum_arg_207(i262):
        # Child args for sum_arg_207
        return(i262*X_coef_cp[i262]*diff(Z_coef_cp[n-i262],'chi',1)+i262*diff(X_coef_cp[i262],'chi',1)*Z_coef_cp[n-i262])
    
    def sum_arg_206(i256):
        # Child args for sum_arg_206
        return(diff(X_coef_cp[i256],'chi',1)*(n-i256)*diff(Z_coef_cp[n-i256],'phi',1)+X_coef_cp[i256]*(n-i256)*diff(Z_coef_cp[n-i256],'chi',1,'phi',1)+diff(X_coef_cp[i256],'phi',1)*(n-i256)*diff(Z_coef_cp[n-i256],'chi',1)+diff(X_coef_cp[i256],'chi',1,'phi',1)*(n-i256)*Z_coef_cp[n-i256])
    
    def sum_arg_205(i256):
        # Child args for sum_arg_205
        return(X_coef_cp[i256]*(n-i256)*diff(Z_coef_cp[n-i256],'chi',1)+diff(X_coef_cp[i256],'chi',1)*(n-i256)*Z_coef_cp[n-i256])
    
    def sum_arg_204(i256):
        # Child args for sum_arg_204
        return(X_coef_cp[i256]*(n-i256)*diff(Z_coef_cp[n-i256],'chi',1)+diff(X_coef_cp[i256],'chi',1)*(n-i256)*Z_coef_cp[n-i256])
    
    def sum_arg_203(i201):
        # Child args for sum_arg_203    
        def sum_arg_202(i202):
            # Child args for sum_arg_202
            return(diff(B_psi_coef_cp[i202],'chi',1,'phi',1)*B_denom_coef_c[(-n)-i202+2*i201+2]*is_seq(n-i201-2,i201-i202)+diff(B_psi_coef_cp[i202],'phi',1)*diff(B_denom_coef_c[(-n)-i202+2*i201+2],'chi',1)*is_seq(n-i201-2,i201-i202))
        
        return(is_seq(0,n-i201-2)*B_alpha_coef[n-i201-2]*is_integer(n-i201-2)*py_sum(sum_arg_202,0,i201))
    
    def sum_arg_201(i292):
        # Child args for sum_arg_201
        return(X_coef_cp[i292]*diff(Y_coef_cp[n-i292],'chi',2)+diff(X_coef_cp[i292],'chi',1)*diff(Y_coef_cp[n-i292],'chi',1))
    
    def sum_arg_200(i288):
        # Child args for sum_arg_200
        return(Y_coef_cp[i288]*diff(X_coef_cp[n-i288],'chi',2)+diff(Y_coef_cp[i288],'chi',1)*diff(X_coef_cp[n-i288],'chi',1))
    
    def sum_arg_199(i294):
        # Child args for sum_arg_199
        return(X_coef_cp[i294]*diff(Z_coef_cp[n-i294],'chi',2)+diff(X_coef_cp[i294],'chi',1)*diff(Z_coef_cp[n-i294],'chi',1))
    
    def sum_arg_198(i290):
        # Child args for sum_arg_198
        return(Z_coef_cp[i290]*diff(X_coef_cp[n-i290],'chi',2)+diff(Z_coef_cp[i290],'chi',1)*diff(X_coef_cp[n-i290],'chi',1))
    
    def sum_arg_197(i285):
        # Child args for sum_arg_197    
        def sum_arg_196(i286):
            # Child args for sum_arg_196
            return(diff(Z_coef_cp[i286],'chi',1)*diff(Z_coef_cp[(-n)-i286+2*i285],'chi',2)*is_seq(n-i285,i285-i286)+diff(Z_coef_cp[i286],'chi',2)*diff(Z_coef_cp[(-n)-i286+2*i285],'chi',1)*is_seq(n-i285,i285-i286))
        
        return(is_seq(0,n-i285)*iota_coef[n-i285]*is_integer(n-i285)*py_sum(sum_arg_196,0,i285))
    
    def sum_arg_195(i284):
        # Child args for sum_arg_195
        return(diff(Z_coef_cp[i284],'chi',2)*diff(Z_coef_cp[n-i284],'phi',1)+diff(Z_coef_cp[i284],'chi',1)*diff(Z_coef_cp[n-i284],'chi',1,'phi',1))
    
    def sum_arg_194(i281):
        # Child args for sum_arg_194    
        def sum_arg_193(i282):
            # Child args for sum_arg_193
            return(diff(Y_coef_cp[i282],'chi',1)*diff(Y_coef_cp[(-n)-i282+2*i281],'chi',2)*is_seq(n-i281,i281-i282)+diff(Y_coef_cp[i282],'chi',2)*diff(Y_coef_cp[(-n)-i282+2*i281],'chi',1)*is_seq(n-i281,i281-i282))
        
        return(is_seq(0,n-i281)*iota_coef[n-i281]*is_integer(n-i281)*py_sum(sum_arg_193,0,i281))
    
    def sum_arg_192(i280):
        # Child args for sum_arg_192
        return(diff(Y_coef_cp[i280],'chi',2)*diff(Y_coef_cp[n-i280],'phi',1)+diff(Y_coef_cp[i280],'chi',1)*diff(Y_coef_cp[n-i280],'chi',1,'phi',1))
    
    def sum_arg_191(i277):
        # Child args for sum_arg_191    
        def sum_arg_190(i278):
            # Child args for sum_arg_190
            return(diff(X_coef_cp[i278],'chi',1)*diff(X_coef_cp[(-n)-i278+2*i277],'chi',2)*is_seq(n-i277,i277-i278)+diff(X_coef_cp[i278],'chi',2)*diff(X_coef_cp[(-n)-i278+2*i277],'chi',1)*is_seq(n-i277,i277-i278))
        
        return(is_seq(0,n-i277)*iota_coef[n-i277]*is_integer(n-i277)*py_sum(sum_arg_190,0,i277))
    
    def sum_arg_189(i276):
        # Child args for sum_arg_189
        return(diff(X_coef_cp[i276],'chi',2)*diff(X_coef_cp[n-i276],'phi',1)+diff(X_coef_cp[i276],'chi',1)*diff(X_coef_cp[n-i276],'chi',1,'phi',1))
    
    def sum_arg_188(i209):
        # Child args for sum_arg_188    
        def sum_arg_187(i210):
            # Child args for sum_arg_187
            return(diff(B_theta_coef_cp[i210],'chi',1)*B_denom_coef_c[(-n)-i210+2*i209]*is_seq(n-i209,i209-i210)+B_theta_coef_cp[i210]*diff(B_denom_coef_c[(-n)-i210+2*i209],'chi',1)*is_seq(n-i209,i209-i210))
        
        return(is_seq(0,n-i209)*B_alpha_coef[n-i209]*is_integer(n-i209)*py_sum(sum_arg_187,0,i209))
    
    def sum_arg_186(i260):
        # Child args for sum_arg_186
        return(i260*X_coef_cp[i260]*diff(Y_coef_cp[n-i260],'chi',2)+2*i260*diff(X_coef_cp[i260],'chi',1)*diff(Y_coef_cp[n-i260],'chi',1)+i260*diff(X_coef_cp[i260],'chi',2)*Y_coef_cp[n-i260])
    
    def sum_arg_185(i258):
        # Child args for sum_arg_185
        return(X_coef_cp[i258]*(n-i258)*diff(Y_coef_cp[n-i258],'chi',2)+2*diff(X_coef_cp[i258],'chi',1)*(n-i258)*diff(Y_coef_cp[n-i258],'chi',1)+diff(X_coef_cp[i258],'chi',2)*(n-i258)*Y_coef_cp[n-i258])
    
    def sum_arg_184(i273):
        # Child args for sum_arg_184    
        def sum_arg_183(i274):
            # Child args for sum_arg_183
            return(i274*Z_coef_cp[i274]*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',3)*is_seq(n-i273,i273-i274)+2*i274*diff(Z_coef_cp[i274],'chi',1)*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',2)*is_seq(n-i273,i273-i274)+i274*diff(Z_coef_cp[i274],'chi',2)*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',1)*is_seq(n-i273,i273-i274))
        
        return(is_seq(0,n-i273)*iota_coef[n-i273]*is_integer(n-i273)*py_sum(sum_arg_183,0,i273))
    
    def sum_arg_182(i271):
        # Child args for sum_arg_182    
        def sum_arg_181(i272):
            # Child args for sum_arg_181
            return(i272*Y_coef_cp[i272]*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',3)*is_seq(n-i271,i271-i272)+2*i272*diff(Y_coef_cp[i272],'chi',1)*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',2)*is_seq(n-i271,i271-i272)+i272*diff(Y_coef_cp[i272],'chi',2)*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',1)*is_seq(n-i271,i271-i272))
        
        return(is_seq(0,n-i271)*iota_coef[n-i271]*is_integer(n-i271)*py_sum(sum_arg_181,0,i271))
    
    def sum_arg_180(i269):
        # Child args for sum_arg_180    
        def sum_arg_179(i270):
            # Child args for sum_arg_179
            return(i270*X_coef_cp[i270]*diff(X_coef_cp[(-n)-i270+2*i269],'chi',3)*is_seq(n-i269,i269-i270)+2*i270*diff(X_coef_cp[i270],'chi',1)*diff(X_coef_cp[(-n)-i270+2*i269],'chi',2)*is_seq(n-i269,i269-i270)+i270*diff(X_coef_cp[i270],'chi',2)*diff(X_coef_cp[(-n)-i270+2*i269],'chi',1)*is_seq(n-i269,i269-i270))
        
        return(is_seq(0,n-i269)*iota_coef[n-i269]*is_integer(n-i269)*py_sum(sum_arg_179,0,i269))
    
    def sum_arg_178(i268):
        # Child args for sum_arg_178
        return(i268*diff(Z_coef_cp[i268],'chi',2)*diff(Z_coef_cp[n-i268],'phi',1)+i268*Z_coef_cp[i268]*diff(Z_coef_cp[n-i268],'chi',2,'phi',1)+2*i268*diff(Z_coef_cp[i268],'chi',1)*diff(Z_coef_cp[n-i268],'chi',1,'phi',1))
    
    def sum_arg_177(i266):
        # Child args for sum_arg_177
        return(i266*diff(Y_coef_cp[i266],'chi',2)*diff(Y_coef_cp[n-i266],'phi',1)+i266*Y_coef_cp[i266]*diff(Y_coef_cp[n-i266],'chi',2,'phi',1)+2*i266*diff(Y_coef_cp[i266],'chi',1)*diff(Y_coef_cp[n-i266],'chi',1,'phi',1))
    
    def sum_arg_176(i264):
        # Child args for sum_arg_176
        return(i264*diff(X_coef_cp[i264],'chi',2)*diff(X_coef_cp[n-i264],'phi',1)+i264*X_coef_cp[i264]*diff(X_coef_cp[n-i264],'chi',2,'phi',1)+2*i264*diff(X_coef_cp[i264],'chi',1)*diff(X_coef_cp[n-i264],'chi',1,'phi',1))
    
    def sum_arg_175(i262):
        # Child args for sum_arg_175
        return(i262*X_coef_cp[i262]*diff(Z_coef_cp[n-i262],'chi',2)+2*i262*diff(X_coef_cp[i262],'chi',1)*diff(Z_coef_cp[n-i262],'chi',1)+i262*diff(X_coef_cp[i262],'chi',2)*Z_coef_cp[n-i262])
    
    def sum_arg_174(i256):
        # Child args for sum_arg_174
        return(X_coef_cp[i256]*(n-i256)*diff(Z_coef_cp[n-i256],'chi',2)+2*diff(X_coef_cp[i256],'chi',1)*(n-i256)*diff(Z_coef_cp[n-i256],'chi',1)+diff(X_coef_cp[i256],'chi',2)*(n-i256)*Z_coef_cp[n-i256])
    
    def sum_arg_173(i201):
        # Child args for sum_arg_173    
        def sum_arg_172(i202):
            # Child args for sum_arg_172
            return(diff(B_psi_coef_cp[i202],'chi',2)*B_denom_coef_c[(-n)-i202+2*i201+2]*is_seq(n-i201-2,i201-i202)+B_psi_coef_cp[i202]*diff(B_denom_coef_c[(-n)-i202+2*i201+2],'chi',2)*is_seq(n-i201-2,i201-i202)+2*diff(B_psi_coef_cp[i202],'chi',1)*diff(B_denom_coef_c[(-n)-i202+2*i201+2],'chi',1)*is_seq(n-i201-2,i201-i202))
        
        return(is_seq(0,n-i201-2)*B_alpha_coef[n-i201-2]*is_integer(n-i201-2)*py_sum(sum_arg_172,0,i201))
    
    def sum_arg_171(i292):
        # Child args for sum_arg_171
        return(X_coef_cp[i292]*diff(Y_coef_cp[n-i292],'chi',1))
    
    def sum_arg_170(i288):
        # Child args for sum_arg_170
        return(Y_coef_cp[i288]*diff(X_coef_cp[n-i288],'chi',1))
    
    def sum_arg_169(i294):
        # Child args for sum_arg_169
        return(X_coef_cp[i294]*diff(Z_coef_cp[n-i294],'chi',1))
    
    def sum_arg_168(i290):
        # Child args for sum_arg_168
        return(Z_coef_cp[i290]*diff(X_coef_cp[n-i290],'chi',1))
    
    def sum_arg_167(i285):
        # Child args for sum_arg_167    
        def sum_arg_166(i286):
            # Child args for sum_arg_166
            return(diff(Z_coef_cp[i286],'chi',1)*diff(Z_coef_cp[(-n)-i286+2*i285],'chi',1)*is_seq(n-i285,i285-i286))
        
        return(is_seq(0,n-i285)*iota_coef[n-i285]*is_integer(n-i285)*py_sum(sum_arg_166,0,i285))
    
    def sum_arg_165(i284):
        # Child args for sum_arg_165
        return(diff(Z_coef_cp[i284],'chi',1)*diff(Z_coef_cp[n-i284],'phi',1))
    
    def sum_arg_164(i281):
        # Child args for sum_arg_164    
        def sum_arg_163(i282):
            # Child args for sum_arg_163
            return(diff(Y_coef_cp[i282],'chi',1)*diff(Y_coef_cp[(-n)-i282+2*i281],'chi',1)*is_seq(n-i281,i281-i282))
        
        return(is_seq(0,n-i281)*iota_coef[n-i281]*is_integer(n-i281)*py_sum(sum_arg_163,0,i281))
    
    def sum_arg_162(i280):
        # Child args for sum_arg_162
        return(diff(Y_coef_cp[i280],'chi',1)*diff(Y_coef_cp[n-i280],'phi',1))
    
    def sum_arg_161(i277):
        # Child args for sum_arg_161    
        def sum_arg_160(i278):
            # Child args for sum_arg_160
            return(diff(X_coef_cp[i278],'chi',1)*diff(X_coef_cp[(-n)-i278+2*i277],'chi',1)*is_seq(n-i277,i277-i278))
        
        return(is_seq(0,n-i277)*iota_coef[n-i277]*is_integer(n-i277)*py_sum(sum_arg_160,0,i277))
    
    def sum_arg_159(i276):
        # Child args for sum_arg_159
        return(diff(X_coef_cp[i276],'chi',1)*diff(X_coef_cp[n-i276],'phi',1))
    
    def sum_arg_158(i209):
        # Child args for sum_arg_158    
        def sum_arg_157(i210):
            # Child args for sum_arg_157
            return(B_theta_coef_cp[i210]*B_denom_coef_c[(-n)-i210+2*i209]*is_seq(n-i209,i209-i210))
        
        return(is_seq(0,n-i209)*B_alpha_coef[n-i209]*is_integer(n-i209)*py_sum(sum_arg_157,0,i209))
    
    def sum_arg_156(i260):
        # Child args for sum_arg_156
        return(i260*X_coef_cp[i260]*diff(Y_coef_cp[n-i260],'chi',1)+i260*diff(X_coef_cp[i260],'chi',1)*Y_coef_cp[n-i260])
    
    def sum_arg_155(i258):
        # Child args for sum_arg_155
        return(X_coef_cp[i258]*(n-i258)*diff(Y_coef_cp[n-i258],'chi',1)+diff(X_coef_cp[i258],'chi',1)*(n-i258)*Y_coef_cp[n-i258])
    
    def sum_arg_154(i273):
        # Child args for sum_arg_154    
        def sum_arg_153(i274):
            # Child args for sum_arg_153
            return(i274*Z_coef_cp[i274]*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',2)*is_seq(n-i273,i273-i274)+i274*diff(Z_coef_cp[i274],'chi',1)*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',1)*is_seq(n-i273,i273-i274))
        
        return(is_seq(0,n-i273)*iota_coef[n-i273]*is_integer(n-i273)*py_sum(sum_arg_153,0,i273))
    
    def sum_arg_152(i271):
        # Child args for sum_arg_152    
        def sum_arg_151(i272):
            # Child args for sum_arg_151
            return(i272*Y_coef_cp[i272]*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',2)*is_seq(n-i271,i271-i272)+i272*diff(Y_coef_cp[i272],'chi',1)*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',1)*is_seq(n-i271,i271-i272))
        
        return(is_seq(0,n-i271)*iota_coef[n-i271]*is_integer(n-i271)*py_sum(sum_arg_151,0,i271))
    
    def sum_arg_150(i269):
        # Child args for sum_arg_150    
        def sum_arg_149(i270):
            # Child args for sum_arg_149
            return(i270*X_coef_cp[i270]*diff(X_coef_cp[(-n)-i270+2*i269],'chi',2)*is_seq(n-i269,i269-i270)+i270*diff(X_coef_cp[i270],'chi',1)*diff(X_coef_cp[(-n)-i270+2*i269],'chi',1)*is_seq(n-i269,i269-i270))
        
        return(is_seq(0,n-i269)*iota_coef[n-i269]*is_integer(n-i269)*py_sum(sum_arg_149,0,i269))
    
    def sum_arg_148(i268):
        # Child args for sum_arg_148
        return(i268*diff(Z_coef_cp[i268],'chi',1)*diff(Z_coef_cp[n-i268],'phi',1)+i268*Z_coef_cp[i268]*diff(Z_coef_cp[n-i268],'chi',1,'phi',1))
    
    def sum_arg_147(i266):
        # Child args for sum_arg_147
        return(i266*diff(Y_coef_cp[i266],'chi',1)*diff(Y_coef_cp[n-i266],'phi',1)+i266*Y_coef_cp[i266]*diff(Y_coef_cp[n-i266],'chi',1,'phi',1))
    
    def sum_arg_146(i264):
        # Child args for sum_arg_146
        return(i264*diff(X_coef_cp[i264],'chi',1)*diff(X_coef_cp[n-i264],'phi',1)+i264*X_coef_cp[i264]*diff(X_coef_cp[n-i264],'chi',1,'phi',1))
    
    def sum_arg_145(i262):
        # Child args for sum_arg_145
        return(i262*X_coef_cp[i262]*diff(Z_coef_cp[n-i262],'chi',1)+i262*diff(X_coef_cp[i262],'chi',1)*Z_coef_cp[n-i262])
    
    def sum_arg_144(i256):
        # Child args for sum_arg_144
        return(X_coef_cp[i256]*(n-i256)*diff(Z_coef_cp[n-i256],'chi',1)+diff(X_coef_cp[i256],'chi',1)*(n-i256)*Z_coef_cp[n-i256])
    
    def sum_arg_143(i201):
        # Child args for sum_arg_143    
        def sum_arg_142(i202):
            # Child args for sum_arg_142
            return(diff(B_psi_coef_cp[i202],'chi',1)*B_denom_coef_c[(-n)-i202+2*i201+2]*is_seq(n-i201-2,i201-i202)+B_psi_coef_cp[i202]*diff(B_denom_coef_c[(-n)-i202+2*i201+2],'chi',1)*is_seq(n-i201-2,i201-i202))
        
        return(is_seq(0,n-i201-2)*B_alpha_coef[n-i201-2]*is_integer(n-i201-2)*py_sum(sum_arg_142,0,i201))
    
    def sum_arg_141(i292):
        # Child args for sum_arg_141
        return(X_coef_cp[i292]*diff(Y_coef_cp[n-i292],'chi',1))
    
    def sum_arg_140(i288):
        # Child args for sum_arg_140
        return(Y_coef_cp[i288]*diff(X_coef_cp[n-i288],'chi',1))
    
    def sum_arg_139(i294):
        # Child args for sum_arg_139
        return(X_coef_cp[i294]*diff(Z_coef_cp[n-i294],'chi',1))
    
    def sum_arg_138(i290):
        # Child args for sum_arg_138
        return(Z_coef_cp[i290]*diff(X_coef_cp[n-i290],'chi',1))
    
    def sum_arg_137(i285):
        # Child args for sum_arg_137    
        def sum_arg_136(i286):
            # Child args for sum_arg_136
            return(diff(Z_coef_cp[i286],'chi',1)*diff(Z_coef_cp[(-n)-i286+2*i285],'chi',1)*is_seq(n-i285,i285-i286))
        
        return(is_seq(0,n-i285)*iota_coef[n-i285]*is_integer(n-i285)*py_sum(sum_arg_136,0,i285))
    
    def sum_arg_135(i284):
        # Child args for sum_arg_135
        return(diff(Z_coef_cp[i284],'chi',1)*diff(Z_coef_cp[n-i284],'phi',1))
    
    def sum_arg_134(i281):
        # Child args for sum_arg_134    
        def sum_arg_133(i282):
            # Child args for sum_arg_133
            return(diff(Y_coef_cp[i282],'chi',1)*diff(Y_coef_cp[(-n)-i282+2*i281],'chi',1)*is_seq(n-i281,i281-i282))
        
        return(is_seq(0,n-i281)*iota_coef[n-i281]*is_integer(n-i281)*py_sum(sum_arg_133,0,i281))
    
    def sum_arg_132(i280):
        # Child args for sum_arg_132
        return(diff(Y_coef_cp[i280],'chi',1)*diff(Y_coef_cp[n-i280],'phi',1))
    
    def sum_arg_131(i277):
        # Child args for sum_arg_131    
        def sum_arg_130(i278):
            # Child args for sum_arg_130
            return(diff(X_coef_cp[i278],'chi',1)*diff(X_coef_cp[(-n)-i278+2*i277],'chi',1)*is_seq(n-i277,i277-i278))
        
        return(is_seq(0,n-i277)*iota_coef[n-i277]*is_integer(n-i277)*py_sum(sum_arg_130,0,i277))
    
    def sum_arg_129(i276):
        # Child args for sum_arg_129
        return(diff(X_coef_cp[i276],'chi',1)*diff(X_coef_cp[n-i276],'phi',1))
    
    def sum_arg_128(i209):
        # Child args for sum_arg_128    
        def sum_arg_127(i210):
            # Child args for sum_arg_127
            return(B_theta_coef_cp[i210]*B_denom_coef_c[(-n)-i210+2*i209]*is_seq(n-i209,i209-i210))
        
        return(is_seq(0,n-i209)*B_alpha_coef[n-i209]*is_integer(n-i209)*py_sum(sum_arg_127,0,i209))
    
    def sum_arg_126(i260):
        # Child args for sum_arg_126
        return(i260*X_coef_cp[i260]*diff(Y_coef_cp[n-i260],'chi',1)+i260*diff(X_coef_cp[i260],'chi',1)*Y_coef_cp[n-i260])
    
    def sum_arg_125(i258):
        # Child args for sum_arg_125
        return(X_coef_cp[i258]*(n-i258)*diff(Y_coef_cp[n-i258],'chi',1)+diff(X_coef_cp[i258],'chi',1)*(n-i258)*Y_coef_cp[n-i258])
    
    def sum_arg_124(i273):
        # Child args for sum_arg_124    
        def sum_arg_123(i274):
            # Child args for sum_arg_123
            return(i274*Z_coef_cp[i274]*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',2)*is_seq(n-i273,i273-i274)+i274*diff(Z_coef_cp[i274],'chi',1)*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',1)*is_seq(n-i273,i273-i274))
        
        return(is_seq(0,n-i273)*iota_coef[n-i273]*is_integer(n-i273)*py_sum(sum_arg_123,0,i273))
    
    def sum_arg_122(i271):
        # Child args for sum_arg_122    
        def sum_arg_121(i272):
            # Child args for sum_arg_121
            return(i272*Y_coef_cp[i272]*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',2)*is_seq(n-i271,i271-i272)+i272*diff(Y_coef_cp[i272],'chi',1)*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',1)*is_seq(n-i271,i271-i272))
        
        return(is_seq(0,n-i271)*iota_coef[n-i271]*is_integer(n-i271)*py_sum(sum_arg_121,0,i271))
    
    def sum_arg_120(i269):
        # Child args for sum_arg_120    
        def sum_arg_119(i270):
            # Child args for sum_arg_119
            return(i270*X_coef_cp[i270]*diff(X_coef_cp[(-n)-i270+2*i269],'chi',2)*is_seq(n-i269,i269-i270)+i270*diff(X_coef_cp[i270],'chi',1)*diff(X_coef_cp[(-n)-i270+2*i269],'chi',1)*is_seq(n-i269,i269-i270))
        
        return(is_seq(0,n-i269)*iota_coef[n-i269]*is_integer(n-i269)*py_sum(sum_arg_119,0,i269))
    
    def sum_arg_118(i268):
        # Child args for sum_arg_118
        return(i268*diff(Z_coef_cp[i268],'chi',1)*diff(Z_coef_cp[n-i268],'phi',1)+i268*Z_coef_cp[i268]*diff(Z_coef_cp[n-i268],'chi',1,'phi',1))
    
    def sum_arg_117(i266):
        # Child args for sum_arg_117
        return(i266*diff(Y_coef_cp[i266],'chi',1)*diff(Y_coef_cp[n-i266],'phi',1)+i266*Y_coef_cp[i266]*diff(Y_coef_cp[n-i266],'chi',1,'phi',1))
    
    def sum_arg_116(i264):
        # Child args for sum_arg_116
        return(i264*diff(X_coef_cp[i264],'chi',1)*diff(X_coef_cp[n-i264],'phi',1)+i264*X_coef_cp[i264]*diff(X_coef_cp[n-i264],'chi',1,'phi',1))
    
    def sum_arg_115(i262):
        # Child args for sum_arg_115
        return(i262*X_coef_cp[i262]*diff(Z_coef_cp[n-i262],'chi',1)+i262*diff(X_coef_cp[i262],'chi',1)*Z_coef_cp[n-i262])
    
    def sum_arg_114(i256):
        # Child args for sum_arg_114
        return(X_coef_cp[i256]*(n-i256)*diff(Z_coef_cp[n-i256],'chi',1)+diff(X_coef_cp[i256],'chi',1)*(n-i256)*Z_coef_cp[n-i256])
    
    def sum_arg_113(i201):
        # Child args for sum_arg_113    
        def sum_arg_112(i202):
            # Child args for sum_arg_112
            return(diff(B_psi_coef_cp[i202],'chi',1)*B_denom_coef_c[(-n)-i202+2*i201+2]*is_seq(n-i201-2,i201-i202)+B_psi_coef_cp[i202]*diff(B_denom_coef_c[(-n)-i202+2*i201+2],'chi',1)*is_seq(n-i201-2,i201-i202))
        
        return(is_seq(0,n-i201-2)*B_alpha_coef[n-i201-2]*is_integer(n-i201-2)*py_sum(sum_arg_112,0,i201))
    
    def sum_arg_111(i292):
        # Child args for sum_arg_111
        return(X_coef_cp[i292]*diff(Y_coef_cp[n-i292],'chi',1))
    
    def sum_arg_110(i288):
        # Child args for sum_arg_110
        return(Y_coef_cp[i288]*diff(X_coef_cp[n-i288],'chi',1))
    
    def sum_arg_109(i294):
        # Child args for sum_arg_109
        return(X_coef_cp[i294]*diff(Z_coef_cp[n-i294],'chi',1))
    
    def sum_arg_108(i290):
        # Child args for sum_arg_108
        return(Z_coef_cp[i290]*diff(X_coef_cp[n-i290],'chi',1))
    
    def sum_arg_107(i285):
        # Child args for sum_arg_107    
        def sum_arg_106(i286):
            # Child args for sum_arg_106
            return(diff(Z_coef_cp[i286],'chi',1)*diff(Z_coef_cp[(-n)-i286+2*i285],'chi',1)*is_seq(n-i285,i285-i286))
        
        return(is_seq(0,n-i285)*iota_coef[n-i285]*is_integer(n-i285)*py_sum(sum_arg_106,0,i285))
    
    def sum_arg_105(i284):
        # Child args for sum_arg_105
        return(diff(Z_coef_cp[i284],'chi',1)*diff(Z_coef_cp[n-i284],'phi',1))
    
    def sum_arg_104(i281):
        # Child args for sum_arg_104    
        def sum_arg_103(i282):
            # Child args for sum_arg_103
            return(diff(Y_coef_cp[i282],'chi',1)*diff(Y_coef_cp[(-n)-i282+2*i281],'chi',1)*is_seq(n-i281,i281-i282))
        
        return(is_seq(0,n-i281)*iota_coef[n-i281]*is_integer(n-i281)*py_sum(sum_arg_103,0,i281))
    
    def sum_arg_102(i280):
        # Child args for sum_arg_102
        return(diff(Y_coef_cp[i280],'chi',1)*diff(Y_coef_cp[n-i280],'phi',1))
    
    def sum_arg_101(i277):
        # Child args for sum_arg_101    
        def sum_arg_100(i278):
            # Child args for sum_arg_100
            return(diff(X_coef_cp[i278],'chi',1)*diff(X_coef_cp[(-n)-i278+2*i277],'chi',1)*is_seq(n-i277,i277-i278))
        
        return(is_seq(0,n-i277)*iota_coef[n-i277]*is_integer(n-i277)*py_sum(sum_arg_100,0,i277))
    
    def sum_arg_99(i276):
        # Child args for sum_arg_99
        return(diff(X_coef_cp[i276],'chi',1)*diff(X_coef_cp[n-i276],'phi',1))
    
    def sum_arg_98(i209):
        # Child args for sum_arg_98    
        def sum_arg_97(i210):
            # Child args for sum_arg_97
            return(B_theta_coef_cp[i210]*B_denom_coef_c[(-n)-i210+2*i209]*is_seq(n-i209,i209-i210))
        
        return(is_seq(0,n-i209)*B_alpha_coef[n-i209]*is_integer(n-i209)*py_sum(sum_arg_97,0,i209))
    
    def sum_arg_96(i260):
        # Child args for sum_arg_96
        return(i260*X_coef_cp[i260]*diff(Y_coef_cp[n-i260],'chi',1)+i260*diff(X_coef_cp[i260],'chi',1)*Y_coef_cp[n-i260])
    
    def sum_arg_95(i258):
        # Child args for sum_arg_95
        return(X_coef_cp[i258]*(n-i258)*diff(Y_coef_cp[n-i258],'chi',1)+diff(X_coef_cp[i258],'chi',1)*(n-i258)*Y_coef_cp[n-i258])
    
    def sum_arg_94(i273):
        # Child args for sum_arg_94    
        def sum_arg_93(i274):
            # Child args for sum_arg_93
            return(i274*Z_coef_cp[i274]*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',2)*is_seq(n-i273,i273-i274)+i274*diff(Z_coef_cp[i274],'chi',1)*diff(Z_coef_cp[(-n)-i274+2*i273],'chi',1)*is_seq(n-i273,i273-i274))
        
        return(is_seq(0,n-i273)*iota_coef[n-i273]*is_integer(n-i273)*py_sum(sum_arg_93,0,i273))
    
    def sum_arg_92(i271):
        # Child args for sum_arg_92    
        def sum_arg_91(i272):
            # Child args for sum_arg_91
            return(i272*Y_coef_cp[i272]*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',2)*is_seq(n-i271,i271-i272)+i272*diff(Y_coef_cp[i272],'chi',1)*diff(Y_coef_cp[(-n)-i272+2*i271],'chi',1)*is_seq(n-i271,i271-i272))
        
        return(is_seq(0,n-i271)*iota_coef[n-i271]*is_integer(n-i271)*py_sum(sum_arg_91,0,i271))
    
    def sum_arg_90(i269):
        # Child args for sum_arg_90    
        def sum_arg_89(i270):
            # Child args for sum_arg_89
            return(i270*X_coef_cp[i270]*diff(X_coef_cp[(-n)-i270+2*i269],'chi',2)*is_seq(n-i269,i269-i270)+i270*diff(X_coef_cp[i270],'chi',1)*diff(X_coef_cp[(-n)-i270+2*i269],'chi',1)*is_seq(n-i269,i269-i270))
        
        return(is_seq(0,n-i269)*iota_coef[n-i269]*is_integer(n-i269)*py_sum(sum_arg_89,0,i269))
    
    def sum_arg_88(i268):
        # Child args for sum_arg_88
        return(i268*diff(Z_coef_cp[i268],'chi',1)*diff(Z_coef_cp[n-i268],'phi',1)+i268*Z_coef_cp[i268]*diff(Z_coef_cp[n-i268],'chi',1,'phi',1))
    
    def sum_arg_87(i266):
        # Child args for sum_arg_87
        return(i266*diff(Y_coef_cp[i266],'chi',1)*diff(Y_coef_cp[n-i266],'phi',1)+i266*Y_coef_cp[i266]*diff(Y_coef_cp[n-i266],'chi',1,'phi',1))
    
    def sum_arg_86(i264):
        # Child args for sum_arg_86
        return(i264*diff(X_coef_cp[i264],'chi',1)*diff(X_coef_cp[n-i264],'phi',1)+i264*X_coef_cp[i264]*diff(X_coef_cp[n-i264],'chi',1,'phi',1))
    
    def sum_arg_85(i262):
        # Child args for sum_arg_85
        return(i262*X_coef_cp[i262]*diff(Z_coef_cp[n-i262],'chi',1)+i262*diff(X_coef_cp[i262],'chi',1)*Z_coef_cp[n-i262])
    
    def sum_arg_84(i256):
        # Child args for sum_arg_84
        return(X_coef_cp[i256]*(n-i256)*diff(Z_coef_cp[n-i256],'chi',1)+diff(X_coef_cp[i256],'chi',1)*(n-i256)*Z_coef_cp[n-i256])
    
    def sum_arg_83(i201):
        # Child args for sum_arg_83    
        def sum_arg_82(i202):
            # Child args for sum_arg_82
            return(diff(B_psi_coef_cp[i202],'chi',1)*B_denom_coef_c[(-n)-i202+2*i201+2]*is_seq(n-i201-2,i201-i202)+B_psi_coef_cp[i202]*diff(B_denom_coef_c[(-n)-i202+2*i201+2],'chi',1)*is_seq(n-i201-2,i201-i202))
        
        return(is_seq(0,n-i201-2)*B_alpha_coef[n-i201-2]*is_integer(n-i201-2)*py_sum(sum_arg_82,0,i201))
    
    def sum_arg_81(i338):
        # Child args for sum_arg_81    
        def sum_arg_79(i300):
            # Child args for sum_arg_79
            return(B_psi_coef_cp[i300]*diff(Delta_coef_cp[n-i338-i300-2],'phi',1))
            
        def sum_arg_80(i300):
            # Child args for sum_arg_80
            return(diff(B_psi_coef_cp[i300],'chi',1)*diff(Delta_coef_cp[n-i338-i300-2],'phi',1)+B_psi_coef_cp[i300]*diff(Delta_coef_cp[n-i338-i300-2],'chi',1,'phi',1))
        
        return(B_denom_coef_c[i338]*py_sum(sum_arg_80,0,n-i338-2)+diff(B_denom_coef_c[i338],'chi',1)*py_sum(sum_arg_79,0,n-i338-2))
    
    def sum_arg_78(i334):
        # Child args for sum_arg_78    
        def sum_arg_76(i332):
            # Child args for sum_arg_76
            return(Delta_coef_cp[i332]*diff(B_psi_coef_cp[n-i334-i332-2],'phi',1))
            
        def sum_arg_77(i332):
            # Child args for sum_arg_77
            return(diff(Delta_coef_cp[i332],'chi',1)*diff(B_psi_coef_cp[n-i334-i332-2],'phi',1)+Delta_coef_cp[i332]*diff(B_psi_coef_cp[n-i334-i332-2],'chi',1,'phi',1))
        
        return(B_denom_coef_c[i334]*py_sum(sum_arg_77,0,n-i334-2)+diff(B_denom_coef_c[i334],'chi',1)*py_sum(sum_arg_76,0,n-i334-2))
    
    def sum_arg_75(i329):
        # Child args for sum_arg_75    
        def sum_arg_74(i330):
            # Child args for sum_arg_74    
            def sum_arg_72(i302):
                # Child args for sum_arg_72
                return(B_psi_coef_cp[i302]*diff(Delta_coef_cp[(-n)-i330+2*i329-i302+2],'chi',1))
                
            def sum_arg_73(i302):
                # Child args for sum_arg_73
                return(B_psi_coef_cp[i302]*diff(Delta_coef_cp[(-n)-i330+2*i329-i302+2],'chi',2)+diff(B_psi_coef_cp[i302],'chi',1)*diff(Delta_coef_cp[(-n)-i330+2*i329-i302+2],'chi',1))
            
            return(B_denom_coef_c[i330]*is_seq(n-i329-2,i329-i330)*py_sum(sum_arg_73,0,(-n)-i330+2*i329+2)+diff(B_denom_coef_c[i330],'chi',1)*is_seq(n-i329-2,i329-i330)*py_sum(sum_arg_72,0,(-n)-i330+2*i329+2))
        
        return(is_seq(0,n-i329-2)*iota_coef[n-i329-2]*is_integer(n-i329-2)*py_sum(sum_arg_74,0,i329))
    
    def sum_arg_71(i325):
        # Child args for sum_arg_71    
        def sum_arg_70(i326):
            # Child args for sum_arg_70    
            def sum_arg_68(i322):
                # Child args for sum_arg_68
                return(Delta_coef_cp[i322]*diff(B_psi_coef_cp[(-n)-i326+2*i325-i322+2],'chi',1))
                
            def sum_arg_69(i322):
                # Child args for sum_arg_69
                return(Delta_coef_cp[i322]*diff(B_psi_coef_cp[(-n)-i326+2*i325-i322+2],'chi',2)+diff(Delta_coef_cp[i322],'chi',1)*diff(B_psi_coef_cp[(-n)-i326+2*i325-i322+2],'chi',1))
            
            return(B_denom_coef_c[i326]*is_seq(n-i325-2,i325-i326)*py_sum(sum_arg_69,0,(-n)-i326+2*i325+2)+diff(B_denom_coef_c[i326],'chi',1)*is_seq(n-i325-2,i325-i326)*py_sum(sum_arg_68,0,(-n)-i326+2*i325+2))
        
        return(is_seq(0,n-i325-2)*iota_coef[n-i325-2]*is_integer(n-i325-2)*py_sum(sum_arg_70,0,i325))
    
    def sum_arg_67(i315):
        # Child args for sum_arg_67    
        def sum_arg_66(i316):
            # Child args for sum_arg_66
            return(diff(B_denom_coef_c[i316],'chi',1)*Delta_coef_cp[(-n)-i316+2*i315]*is_seq(n-i315,i315-i316)+B_denom_coef_c[i316]*diff(Delta_coef_cp[(-n)-i316+2*i315],'chi',1)*is_seq(n-i315,i315-i316))
        
        return(is_seq(0,n-i315)*(n-i315)*B_alpha_coef[n-i315]*is_integer(n-i315)*py_sum(sum_arg_66,0,i315))
    
    def sum_arg_65(i309):
        # Child args for sum_arg_65    
        def sum_arg_64(i310):
            # Child args for sum_arg_64
            return(B_denom_coef_c[i310]*diff(B_psi_coef_cp[(-n)-i310+2*i309+2],'chi',2)*is_seq(n-i309-2,i309-i310)+diff(B_denom_coef_c[i310],'chi',1)*diff(B_psi_coef_cp[(-n)-i310+2*i309+2],'chi',1)*is_seq(n-i309-2,i309-i310))
        
        return(is_seq(0,n-i309-2)*iota_coef[n-i309-2]*is_integer(n-i309-2)*py_sum(sum_arg_64,0,i309))
    
    def sum_arg_63(i308):
        # Child args for sum_arg_63
        return(diff(B_denom_coef_c[i308],'chi',1)*diff(B_psi_coef_cp[n-i308-2],'phi',1)+B_denom_coef_c[i308]*diff(B_psi_coef_cp[n-i308-2],'chi',1,'phi',1))
    
    def sum_arg_62(i303):
        # Child args for sum_arg_62    
        def sum_arg_61(i304):
            # Child args for sum_arg_61
            return(i304*diff(B_denom_coef_c[i304],'chi',1)*Delta_coef_cp[(-n)-i304+2*i303]*is_seq(n-i303,i303-i304)+i304*B_denom_coef_c[i304]*diff(Delta_coef_cp[(-n)-i304+2*i303],'chi',1)*is_seq(n-i303,i303-i304))
        
        return(is_seq(0,n-i303)*B_alpha_coef[n-i303]*is_integer(n-i303)*py_sum(sum_arg_61,0,i303))
    
    def sum_arg_60(i295):
        # Child args for sum_arg_60    
        def sum_arg_59(i296):
            # Child args for sum_arg_59    
            def sum_arg_57(i250):
                # Child args for sum_arg_57
                return(B_denom_coef_c[i250]*B_denom_coef_c[(-n)-i296+2*i295-i250])
                
            def sum_arg_58(i250):
                # Child args for sum_arg_58
                return(diff(B_denom_coef_c[i250],'chi',1)*B_denom_coef_c[(-n)-i296+2*i295-i250]+B_denom_coef_c[i250]*diff(B_denom_coef_c[(-n)-i296+2*i295-i250],'chi',1))
            
            return(i296*p_perp_coef_cp[i296]*is_seq(n-i295,i295-i296)*py_sum(sum_arg_58,0,(-n)-i296+2*i295)+i296*diff(p_perp_coef_cp[i296],'chi',1)*is_seq(n-i295,i295-i296)*py_sum(sum_arg_57,0,(-n)-i296+2*i295))
        
        return(is_seq(0,n-i295)*B_alpha_coef[n-i295]*is_integer(n-i295)*py_sum(sum_arg_59,0,i295))
    
    def sum_arg_56(i335):
        # Child args for sum_arg_56    
        def sum_arg_55(i336):
            # Child args for sum_arg_55
            return(is_seq(0,(-n)-i336+2*i335)*diff(B_denom_coef_c[i336],'chi',1)*B_theta_coef_cp[(-n)-i336+2*i335]*is_integer((-n)-i336+2*i335)*is_seq((-n)-i336+2*i335,i335-i336)+is_seq(0,(-n)-i336+2*i335)*B_denom_coef_c[i336]*diff(B_theta_coef_cp[(-n)-i336+2*i335],'chi',1)*is_integer((-n)-i336+2*i335)*is_seq((-n)-i336+2*i335,i335-i336))
        
        return((n-i335)*iota_coef[n-i335]*py_sum(sum_arg_55,0,i335))
    
    def sum_arg_54(i319):
        # Child args for sum_arg_54    
        def sum_arg_53(i320):
            # Child args for sum_arg_53    
            def sum_arg_51(i318):
                # Child args for sum_arg_51
                return(is_seq(0,(-n)-i320+2*i319-i318)*Delta_coef_cp[i318]*B_theta_coef_cp[(-n)-i320+2*i319-i318]*is_integer((-n)-i320+2*i319-i318)*is_seq((-n)-i320+2*i319-i318,(-i320)+i319-i318))
                
            def sum_arg_52(i318):
                # Child args for sum_arg_52
                return(is_seq(0,(-n)-i320+2*i319-i318)*diff(Delta_coef_cp[i318],'chi',1)*B_theta_coef_cp[(-n)-i320+2*i319-i318]*is_integer((-n)-i320+2*i319-i318)*is_seq((-n)-i320+2*i319-i318,(-i320)+i319-i318)+is_seq(0,(-n)-i320+2*i319-i318)*Delta_coef_cp[i318]*diff(B_theta_coef_cp[(-n)-i320+2*i319-i318],'chi',1)*is_integer((-n)-i320+2*i319-i318)*is_seq((-n)-i320+2*i319-i318,(-i320)+i319-i318))
            
            return(B_denom_coef_c[i320]*py_sum(sum_arg_52,0,i319-i320)+diff(B_denom_coef_c[i320],'chi',1)*py_sum(sum_arg_51,0,i319-i320))
        
        return((n-i319)*iota_coef[n-i319]*py_sum(sum_arg_53,0,i319))
    
    def sum_arg_50(i311):
        # Child args for sum_arg_50
        return((is_seq(0,n-i311))\
            *(diff(B_denom_coef_c[2*i311-n],'chi',1))\
            *(n-i311)\
            *(B_alpha_coef[n-i311])\
            *(is_integer(n-i311))\
            *(is_seq(n-i311,i311)))
    
    def sum_arg_49(i338):
        # Child args for sum_arg_49    
        def sum_arg_48(i300):
            # Child args for sum_arg_48
            return(B_psi_coef_cp[i300]*diff(Delta_coef_cp[n-i338-i300-2],'phi',1))
        
        return(B_denom_coef_c[i338]*py_sum(sum_arg_48,0,n-i338-2))
    
    def sum_arg_47(i334):
        # Child args for sum_arg_47    
        def sum_arg_46(i332):
            # Child args for sum_arg_46
            return(Delta_coef_cp[i332]*diff(B_psi_coef_cp[n-i334-i332-2],'phi',1))
        
        return(B_denom_coef_c[i334]*py_sum(sum_arg_46,0,n-i334-2))
    
    def sum_arg_45(i329):
        # Child args for sum_arg_45    
        def sum_arg_44(i330):
            # Child args for sum_arg_44    
            def sum_arg_43(i302):
                # Child args for sum_arg_43
                return(B_psi_coef_cp[i302]*diff(Delta_coef_cp[(-n)-i330+2*i329-i302+2],'chi',1))
            
            return(B_denom_coef_c[i330]*is_seq(n-i329-2,i329-i330)*py_sum(sum_arg_43,0,(-n)-i330+2*i329+2))
        
        return(is_seq(0,n-i329-2)*iota_coef[n-i329-2]*is_integer(n-i329-2)*py_sum(sum_arg_44,0,i329))
    
    def sum_arg_42(i325):
        # Child args for sum_arg_42    
        def sum_arg_41(i326):
            # Child args for sum_arg_41    
            def sum_arg_40(i322):
                # Child args for sum_arg_40
                return(Delta_coef_cp[i322]*diff(B_psi_coef_cp[(-n)-i326+2*i325-i322+2],'chi',1))
            
            return(B_denom_coef_c[i326]*is_seq(n-i325-2,i325-i326)*py_sum(sum_arg_40,0,(-n)-i326+2*i325+2))
        
        return(is_seq(0,n-i325-2)*iota_coef[n-i325-2]*is_integer(n-i325-2)*py_sum(sum_arg_41,0,i325))
    
    def sum_arg_39(i315):
        # Child args for sum_arg_39    
        def sum_arg_38(i316):
            # Child args for sum_arg_38
            return(B_denom_coef_c[i316]*Delta_coef_cp[(-n)-i316+2*i315]*is_seq(n-i315,i315-i316))
        
        return(is_seq(0,n-i315)*(n-i315)*B_alpha_coef[n-i315]*is_integer(n-i315)*py_sum(sum_arg_38,0,i315))
    
    def sum_arg_37(i309):
        # Child args for sum_arg_37    
        def sum_arg_36(i310):
            # Child args for sum_arg_36
            return(B_denom_coef_c[i310]*diff(B_psi_coef_cp[(-n)-i310+2*i309+2],'chi',1)*is_seq(n-i309-2,i309-i310))
        
        return(is_seq(0,n-i309-2)*iota_coef[n-i309-2]*is_integer(n-i309-2)*py_sum(sum_arg_36,0,i309))
    
    def sum_arg_35(i308):
        # Child args for sum_arg_35
        return(B_denom_coef_c[i308]*diff(B_psi_coef_cp[n-i308-2],'phi',1))
    
    def sum_arg_34(i303):
        # Child args for sum_arg_34    
        def sum_arg_33(i304):
            # Child args for sum_arg_33
            return(i304*B_denom_coef_c[i304]*Delta_coef_cp[(-n)-i304+2*i303]*is_seq(n-i303,i303-i304))
        
        return(is_seq(0,n-i303)*B_alpha_coef[n-i303]*is_integer(n-i303)*py_sum(sum_arg_33,0,i303))
    
    def sum_arg_32(i295):
        # Child args for sum_arg_32    
        def sum_arg_31(i296):
            # Child args for sum_arg_31    
            def sum_arg_30(i250):
                # Child args for sum_arg_30
                return(B_denom_coef_c[i250]*B_denom_coef_c[(-n)-i296+2*i295-i250])
            
            return(i296*p_perp_coef_cp[i296]*is_seq(n-i295,i295-i296)*py_sum(sum_arg_30,0,(-n)-i296+2*i295))
        
        return(is_seq(0,n-i295)*B_alpha_coef[n-i295]*is_integer(n-i295)*py_sum(sum_arg_31,0,i295))
    
    def sum_arg_29(i335):
        # Child args for sum_arg_29    
        def sum_arg_28(i336):
            # Child args for sum_arg_28
            return(is_seq(0,(-n)-i336+2*i335)*B_denom_coef_c[i336]*B_theta_coef_cp[(-n)-i336+2*i335]*is_integer((-n)-i336+2*i335)*is_seq((-n)-i336+2*i335,i335-i336))
        
        return((n-i335)*iota_coef[n-i335]*py_sum(sum_arg_28,0,i335))
    
    def sum_arg_27(i319):
        # Child args for sum_arg_27    
        def sum_arg_26(i320):
            # Child args for sum_arg_26    
            def sum_arg_25(i318):
                # Child args for sum_arg_25
                return(is_seq(0,(-n)-i320+2*i319-i318)*Delta_coef_cp[i318]*B_theta_coef_cp[(-n)-i320+2*i319-i318]*is_integer((-n)-i320+2*i319-i318)*is_seq((-n)-i320+2*i319-i318,(-i320)+i319-i318))
            
            return(B_denom_coef_c[i320]*py_sum(sum_arg_25,0,i319-i320))
        
        return((n-i319)*iota_coef[n-i319]*py_sum(sum_arg_26,0,i319))
    
    def sum_arg_24(i311):
        # Child args for sum_arg_24
        return((is_seq(0,n-i311))\
            *(B_denom_coef_c[2*i311-n])\
            *(n-i311)\
            *(B_alpha_coef[n-i311])\
            *(is_integer(n-i311))\
            *(is_seq(n-i311,i311)))
    
    def sum_arg_23(i817):
        # Child args for sum_arg_23    
        def sum_arg_22(i818):
            # Child args for sum_arg_22    
            def sum_arg_21(i816):
                # Child args for sum_arg_21
                return(Delta_coef_cp[i816]*diff(B_theta_coef_cp[(-n)-i818+2*i817-i816],'chi',1)*is_seq(n-i817,(-i818)+i817-i816))
            
            return(B_denom_coef_c[i818]*py_sum(sum_arg_21,0,i817-i818))
        
        return(is_seq(0,n-i817)*iota_coef[n-i817]*is_integer(n-i817)*py_sum(sum_arg_22,0,i817))
    
    def sum_arg_20(i814):
        # Child args for sum_arg_20    
        def sum_arg_19(i812):
            # Child args for sum_arg_19
            return(Delta_coef_cp[i812]*diff(B_theta_coef_cp[n-i814-i812],'phi',1))
        
        return(B_denom_coef_c[i814]*py_sum(sum_arg_19,0,n-i814))
    
    def sum_arg_18(i230):
        # Child args for sum_arg_18    
        def sum_arg_17(i228):
            # Child args for sum_arg_17    
            def sum_arg_16(i226):
                # Child args for sum_arg_16
                return(B_denom_coef_c[i226]*B_denom_coef_c[n-i230-i228-i226])
            
            return(B_theta_coef_cp[i228]*py_sum(sum_arg_16,0,n-i230-i228))
        
        return(diff(p_perp_coef_cp[i230],'phi',1)*py_sum(sum_arg_17,0,n-i230))
    
    def sum_arg_15(i809):
        # Child args for sum_arg_15    
        def sum_arg_14(i810):
            # Child args for sum_arg_14
            return(B_denom_coef_c[i810]*diff(B_theta_coef_cp[(-n)-i810+2*i809],'chi',1)*is_seq(n-i809,i809-i810))
        
        return(is_seq(0,n-i809)*iota_coef[n-i809]*is_integer(n-i809)*py_sum(sum_arg_14,0,i809))
    
    def sum_arg_13(i808):
        # Child args for sum_arg_13
        return(B_denom_coef_c[i808]*diff(B_theta_coef_cp[n-i808],'phi',1))
    
    def sum_arg_12(i825):
        # Child args for sum_arg_12    
        def sum_arg_11(i826):
            # Child args for sum_arg_11    
            def sum_arg_10(i1273):
                # Child args for sum_arg_10
                return(Delta_coef_cp[i1273]*diff(B_denom_coef_c[(-i826)+i825-i1273],'chi',1))
            
            return(is_seq(0,(-n)+i826+i825)*B_theta_coef_cp[(-n)+i826+i825]*is_integer((-n)+i826+i825)*is_seq((-n)+i826+i825,i826)*py_sum(sum_arg_10,0,i825-i826))
        
        return(iota_coef[n-i825]*py_sum(sum_arg_11,0,i825))
    
    def sum_arg_9(i823):
        # Child args for sum_arg_9    
        def sum_arg_8(i240):
            # Child args for sum_arg_8
            return(Delta_coef_cp[i240]*diff(B_denom_coef_c[(-n)+2*i823-i240],'chi',1))
        
        return(is_seq(0,n-i823)*B_alpha_coef[n-i823]*is_integer(n-i823)*is_seq(n-i823,i823)*py_sum(sum_arg_8,0,2*i823-n))
    
    def sum_arg_7(i821):
        # Child args for sum_arg_7    
        def sum_arg_6(i822):
            # Child args for sum_arg_6    
            def sum_arg_5(i1235):
                # Child args for sum_arg_5    
                def sum_arg_4(i1257):
                    # Child args for sum_arg_4
                    return(B_denom_coef_c[i1257]*B_denom_coef_c[(-i822)+i821-i1257-i1235])
                
                return(diff(p_perp_coef_cp[i1235],'chi',1)*py_sum(sum_arg_4,0,(-i822)+i821-i1235))
            
            return(is_seq(0,(-n)+i822+i821)*B_theta_coef_cp[(-n)+i822+i821]*is_integer((-n)+i822+i821)*is_seq((-n)+i822+i821,i822)*py_sum(sum_arg_5,0,i821-i822))
        
        return(iota_coef[n-i821]*py_sum(sum_arg_6,0,i821))
    
    def sum_arg_3(i819):
        # Child args for sum_arg_3    
        def sum_arg_2(i236):
            # Child args for sum_arg_2    
            def sum_arg_1(i232):
                # Child args for sum_arg_1
                return(B_denom_coef_c[i232]*B_denom_coef_c[(-n)+2*i819-i236-i232])
            
            return(diff(p_perp_coef_cp[i236],'chi',1)*py_sum(sum_arg_1,0,(-n)+2*i819-i236))
        
        return(is_seq(0,n-i819)*B_alpha_coef[n-i819]*is_integer(n-i819)*is_seq(n-i819,i819)*py_sum(sum_arg_2,0,2*i819-n))
    
    
    out = (B_alpha_coef[0]*B_denom_coef_c[0]**2*(((2*(Delta_coef_cp[0]-1)*diff(B_denom_coef_c[0],'chi',1))/(B_alpha_coef[0]*B_denom_coef_c[0]**2*n)-(2*diff(Delta_coef_cp[0],'chi',1))/(B_alpha_coef[0]*B_denom_coef_c[0]*n))*int_chi(((n*(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_323,0,n)*diff(tau_p,'phi',1)-is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_322,0,n)*diff(tau_p,'phi',1)+is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_321,0,n)*tau_p+is_seq(0,n)*diff(dl_p,'phi',1)*is_integer(n)*py_sum_parallel(sum_arg_320,0,n)*tau_p-is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_319,0,n)*tau_p-is_seq(0,n)*diff(dl_p,'phi',1)*is_integer(n)*py_sum_parallel(sum_arg_318,0,n)*tau_p+is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_317,0,n)+is_seq(0,n)*dl_p*diff(kap_p,'phi',1)*is_integer(n)*py_sum_parallel(sum_arg_316,0,n)+is_seq(0,n)*diff(dl_p,'phi',1)*kap_p*is_integer(n)*py_sum_parallel(sum_arg_315,0,n)-is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_314,0,n)-is_seq(0,n)*dl_p*diff(kap_p,'phi',1)*is_integer(n)*py_sum_parallel(sum_arg_313,0,n)-is_seq(0,n)*diff(dl_p,'phi',1)*kap_p*is_integer(n)*py_sum_parallel(sum_arg_312,0,n)-py_sum_parallel(sum_arg_311,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_309,0,n)-py_sum_parallel(sum_arg_308,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_306,0,n)-py_sum_parallel(sum_arg_305,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_303,0,n)+py_sum_parallel(sum_arg_302,ceil(0.5*n),floor(n))-is_seq(0,n)*dl_p*is_integer(n)*diff(Z_coef_cp[n],'chi',1,'phi',1)-is_seq(0,n)*diff(dl_p,'phi',1)*is_integer(n)*diff(Z_coef_cp[n],'chi',1)))/2+(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_300,0,n)*diff(tau_p,'phi',1))/2-(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_299,0,n)*diff(tau_p,'phi',1))/2+(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_298,0,n)*tau_p)/2+(is_seq(0,n)*diff(dl_p,'phi',1)*is_integer(n)*py_sum_parallel(sum_arg_297,0,n)*tau_p)/2-(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_296,0,n)*tau_p)/2-(is_seq(0,n)*diff(dl_p,'phi',1)*is_integer(n)*py_sum_parallel(sum_arg_295,0,n)*tau_p)/2+py_sum_parallel(sum_arg_294,ceil(0.5*n),floor(n))/2+py_sum_parallel(sum_arg_292,ceil(0.5*n),floor(n))/2+py_sum_parallel(sum_arg_290,ceil(0.5*n),floor(n))/2+(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_288,0,n))/2+(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_287,0,n))/2+(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_286,0,n))/2+(is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_285,0,n))/2+(is_seq(0,n)*dl_p*diff(kap_p,'phi',1)*is_integer(n)*py_sum_parallel(sum_arg_284,0,n))/2+(is_seq(0,n)*diff(dl_p,'phi',1)*kap_p*is_integer(n)*py_sum_parallel(sum_arg_283,0,n))/2-(is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_282,0,n))/2-(is_seq(0,n)*dl_p*diff(kap_p,'phi',1)*is_integer(n)*py_sum_parallel(sum_arg_281,0,n))/2-(is_seq(0,n)*diff(dl_p,'phi',1)*kap_p*is_integer(n)*py_sum_parallel(sum_arg_280,0,n))/2-py_sum_parallel(sum_arg_279,ceil(0.5*n)-1,floor(n)-2)+(is_seq(0,n)*dl_p*n*is_integer(n)*diff(Z_coef_cp[n],'chi',1,'phi',1))/2+(is_seq(0,n)*diff(dl_p,'phi',1)*n*is_integer(n)*diff(Z_coef_cp[n],'chi',1))/2)/(B_alpha_coef[0]*B_denom_coef_c[0]))+((2*diff(B_denom_coef_c[0],'chi',1)*(diff(Delta_coef_cp[0],'phi',1)+iota_coef[0]*diff(Delta_coef_cp[0],'chi',1)))/(B_alpha_coef[0]*B_denom_coef_c[0]**2*n)-(2*(iota_coef[0]*diff(Delta_coef_cp[0],'chi',2)+diff(Delta_coef_cp[0],'chi',1,'phi',1)))/(B_alpha_coef[0]*B_denom_coef_c[0]*n))*int_chi(((n*(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_277,0,n)*tau_p-is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_276,0,n)*tau_p+is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_275,0,n)-is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_274,0,n)-py_sum_parallel(sum_arg_273,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_271,0,n)-py_sum_parallel(sum_arg_270,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_268,0,n)-py_sum_parallel(sum_arg_267,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_265,0,n)+py_sum_parallel(sum_arg_264,ceil(0.5*n),floor(n))-is_seq(0,n)*dl_p*is_integer(n)*diff(Z_coef_cp[n],'chi',1)))/2+(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_262,0,n)*tau_p)/2-(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_261,0,n)*tau_p)/2+py_sum_parallel(sum_arg_260,ceil(0.5*n),floor(n))/2+py_sum_parallel(sum_arg_258,ceil(0.5*n),floor(n))/2+py_sum_parallel(sum_arg_256,ceil(0.5*n),floor(n))/2+(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_254,0,n))/2+(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_253,0,n))/2+(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_252,0,n))/2+(is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_251,0,n))/2-(is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_250,0,n))/2-py_sum_parallel(sum_arg_249,ceil(0.5*n)-1,floor(n)-2)+(is_seq(0,n)*dl_p*n*is_integer(n)*diff(Z_coef_cp[n],'chi',1))/2)/(B_alpha_coef[0]*B_denom_coef_c[0]))-(2*(Delta_coef_cp[0]-1)*((n*(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_247,0,n)*diff(tau_p,'phi',1)-is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_246,0,n)*diff(tau_p,'phi',1)+is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_245,0,n)*tau_p+is_seq(0,n)*diff(dl_p,'phi',1)*is_integer(n)*py_sum_parallel(sum_arg_244,0,n)*tau_p-is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_243,0,n)*tau_p-is_seq(0,n)*diff(dl_p,'phi',1)*is_integer(n)*py_sum_parallel(sum_arg_242,0,n)*tau_p+is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_241,0,n)+is_seq(0,n)*dl_p*diff(kap_p,'phi',1)*is_integer(n)*py_sum_parallel(sum_arg_240,0,n)+is_seq(0,n)*diff(dl_p,'phi',1)*kap_p*is_integer(n)*py_sum_parallel(sum_arg_239,0,n)-is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_238,0,n)-is_seq(0,n)*dl_p*diff(kap_p,'phi',1)*is_integer(n)*py_sum_parallel(sum_arg_237,0,n)-is_seq(0,n)*diff(dl_p,'phi',1)*kap_p*is_integer(n)*py_sum_parallel(sum_arg_236,0,n)-py_sum_parallel(sum_arg_235,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_233,0,n)-py_sum_parallel(sum_arg_232,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_230,0,n)-py_sum_parallel(sum_arg_229,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_227,0,n)+py_sum_parallel(sum_arg_226,ceil(0.5*n),floor(n))-is_seq(0,n)*dl_p*is_integer(n)*diff(Z_coef_cp[n],'chi',1,'phi',1)-is_seq(0,n)*diff(dl_p,'phi',1)*is_integer(n)*diff(Z_coef_cp[n],'chi',1)))/2+(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_224,0,n)*diff(tau_p,'phi',1))/2-(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_223,0,n)*diff(tau_p,'phi',1))/2+(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_222,0,n)*tau_p)/2+(is_seq(0,n)*diff(dl_p,'phi',1)*is_integer(n)*py_sum_parallel(sum_arg_221,0,n)*tau_p)/2-(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_220,0,n)*tau_p)/2-(is_seq(0,n)*diff(dl_p,'phi',1)*is_integer(n)*py_sum_parallel(sum_arg_219,0,n)*tau_p)/2+py_sum_parallel(sum_arg_218,ceil(0.5*n),floor(n))/2+py_sum_parallel(sum_arg_216,ceil(0.5*n),floor(n))/2+py_sum_parallel(sum_arg_214,ceil(0.5*n),floor(n))/2+(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_212,0,n))/2+(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_211,0,n))/2+(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_210,0,n))/2+(is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_209,0,n))/2+(is_seq(0,n)*dl_p*diff(kap_p,'phi',1)*is_integer(n)*py_sum_parallel(sum_arg_208,0,n))/2+(is_seq(0,n)*diff(dl_p,'phi',1)*kap_p*is_integer(n)*py_sum_parallel(sum_arg_207,0,n))/2-(is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_206,0,n))/2-(is_seq(0,n)*dl_p*diff(kap_p,'phi',1)*is_integer(n)*py_sum_parallel(sum_arg_205,0,n))/2-(is_seq(0,n)*diff(dl_p,'phi',1)*kap_p*is_integer(n)*py_sum_parallel(sum_arg_204,0,n))/2-py_sum_parallel(sum_arg_203,ceil(0.5*n)-1,floor(n)-2)+(is_seq(0,n)*dl_p*n*is_integer(n)*diff(Z_coef_cp[n],'chi',1,'phi',1))/2+(is_seq(0,n)*diff(dl_p,'phi',1)*n*is_integer(n)*diff(Z_coef_cp[n],'chi',1))/2))/(B_alpha_coef[0]**2*B_denom_coef_c[0]**2*n)-(2*(Delta_coef_cp[0]-1)*iota_coef[0]*((n*(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_201,0,n)*tau_p-is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_200,0,n)*tau_p+is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_199,0,n)-is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_198,0,n)-py_sum_parallel(sum_arg_197,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_195,0,n)-py_sum_parallel(sum_arg_194,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_192,0,n)-py_sum_parallel(sum_arg_191,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_189,0,n)+py_sum_parallel(sum_arg_188,ceil(0.5*n),floor(n))-is_seq(0,n)*dl_p*is_integer(n)*diff(Z_coef_cp[n],'chi',2)))/2+(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_186,0,n)*tau_p)/2-(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_185,0,n)*tau_p)/2+py_sum_parallel(sum_arg_184,ceil(0.5*n),floor(n))/2+py_sum_parallel(sum_arg_182,ceil(0.5*n),floor(n))/2+py_sum_parallel(sum_arg_180,ceil(0.5*n),floor(n))/2+(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_178,0,n))/2+(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_177,0,n))/2+(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_176,0,n))/2+(is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_175,0,n))/2-(is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_174,0,n))/2-py_sum_parallel(sum_arg_173,ceil(0.5*n)-1,floor(n)-2)+(is_seq(0,n)*dl_p*n*is_integer(n)*diff(Z_coef_cp[n],'chi',2))/2))/(B_alpha_coef[0]**2*B_denom_coef_c[0]**2*n)-(2*(diff(Delta_coef_cp[0],'phi',1)+iota_coef[0]*diff(Delta_coef_cp[0],'chi',1))*((n*(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_171,0,n)*tau_p-is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_170,0,n)*tau_p+is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_169,0,n)-is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_168,0,n)-py_sum_parallel(sum_arg_167,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_165,0,n)-py_sum_parallel(sum_arg_164,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_162,0,n)-py_sum_parallel(sum_arg_161,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_159,0,n)+py_sum_parallel(sum_arg_158,ceil(0.5*n),floor(n))-is_seq(0,n)*dl_p*is_integer(n)*diff(Z_coef_cp[n],'chi',1)))/2+(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_156,0,n)*tau_p)/2-(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_155,0,n)*tau_p)/2+py_sum_parallel(sum_arg_154,ceil(0.5*n),floor(n))/2+py_sum_parallel(sum_arg_152,ceil(0.5*n),floor(n))/2+py_sum_parallel(sum_arg_150,ceil(0.5*n),floor(n))/2+(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_148,0,n))/2+(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_147,0,n))/2+(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_146,0,n))/2+(is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_145,0,n))/2-(is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_144,0,n))/2-py_sum_parallel(sum_arg_143,ceil(0.5*n)-1,floor(n)-2)+(is_seq(0,n)*dl_p*n*is_integer(n)*diff(Z_coef_cp[n],'chi',1))/2))/(B_alpha_coef[0]**2*B_denom_coef_c[0]**2*n)-(2*iota_coef[0]*diff(Delta_coef_cp[0],'chi',1)*((n*(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_141,0,n)*tau_p-is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_140,0,n)*tau_p+is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_139,0,n)-is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_138,0,n)-py_sum_parallel(sum_arg_137,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_135,0,n)-py_sum_parallel(sum_arg_134,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_132,0,n)-py_sum_parallel(sum_arg_131,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_129,0,n)+py_sum_parallel(sum_arg_128,ceil(0.5*n),floor(n))-is_seq(0,n)*dl_p*is_integer(n)*diff(Z_coef_cp[n],'chi',1)))/2+(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_126,0,n)*tau_p)/2-(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_125,0,n)*tau_p)/2+py_sum_parallel(sum_arg_124,ceil(0.5*n),floor(n))/2+py_sum_parallel(sum_arg_122,ceil(0.5*n),floor(n))/2+py_sum_parallel(sum_arg_120,ceil(0.5*n),floor(n))/2+(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_118,0,n))/2+(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_117,0,n))/2+(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_116,0,n))/2+(is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_115,0,n))/2-(is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_114,0,n))/2-py_sum_parallel(sum_arg_113,ceil(0.5*n)-1,floor(n)-2)+(is_seq(0,n)*dl_p*n*is_integer(n)*diff(Z_coef_cp[n],'chi',1))/2))/(B_alpha_coef[0]**2*B_denom_coef_c[0]**2*n)+(4*(Delta_coef_cp[0]-1)*iota_coef[0]*diff(B_denom_coef_c[0],'chi',1)*((n*(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_111,0,n)*tau_p-is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_110,0,n)*tau_p-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_99,0,n)+py_sum_parallel(sum_arg_98,ceil(0.5*n),floor(n))+is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_109,0,n)-is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_108,0,n)-py_sum_parallel(sum_arg_107,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_105,0,n)-py_sum_parallel(sum_arg_104,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_102,0,n)-py_sum_parallel(sum_arg_101,ceil(0.5*n),floor(n))-is_seq(0,n)*dl_p*is_integer(n)*diff(Z_coef_cp[n],'chi',1)))/2+(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_96,0,n)*tau_p)/2-(is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_95,0,n)*tau_p)/2+py_sum_parallel(sum_arg_94,ceil(0.5*n),floor(n))/2+py_sum_parallel(sum_arg_92,ceil(0.5*n),floor(n))/2+py_sum_parallel(sum_arg_90,ceil(0.5*n),floor(n))/2+(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_88,0,n))/2+(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_87,0,n))/2+(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_86,0,n))/2+(is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_85,0,n))/2-(is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_84,0,n))/2-py_sum_parallel(sum_arg_83,ceil(0.5*n)-1,floor(n)-2)+(is_seq(0,n)*dl_p*n*is_integer(n)*diff(Z_coef_cp[n],'chi',1))/2))/(B_alpha_coef[0]**2*B_denom_coef_c[0]**3*n)+(2*((-is_seq(0,n-2)*is_integer(n-2)*py_sum_parallel(sum_arg_81,0,n-2))-is_seq(0,n-2)*is_integer(n-2)*py_sum_parallel(sum_arg_78,0,n-2)-py_sum_parallel(sum_arg_75,ceil(0.5*n)-1,floor(n)-2)-py_sum_parallel(sum_arg_71,ceil(0.5*n)-1,floor(n)-2)+py_sum_parallel(sum_arg_67,ceil(0.5*n),floor(n))+py_sum_parallel(sum_arg_65,ceil(0.5*n)-1,floor(n)-2)+is_seq(0,n-2)*is_integer(n-2)*py_sum_parallel(sum_arg_63,0,n-2)+py_sum_parallel(sum_arg_62,ceil(0.5*n),floor(n))/4-py_sum_parallel(sum_arg_60,ceil(0.5*n),floor(n))/2+py_sum_parallel(sum_arg_56,ceil(0.5*n),floor(n))-py_sum_parallel(sum_arg_54,ceil(0.5*n),floor(n))-py_sum_parallel(sum_arg_50,ceil(0.5*n),floor(n))))/(B_alpha_coef[0]*B_denom_coef_c[0]**2*n)-(4*diff(B_denom_coef_c[0],'chi',1)*((-is_seq(0,n-2)*is_integer(n-2)*py_sum_parallel(sum_arg_49,0,n-2))-is_seq(0,n-2)*is_integer(n-2)*py_sum_parallel(sum_arg_47,0,n-2)-py_sum_parallel(sum_arg_45,ceil(0.5*n)-1,floor(n)-2)-py_sum_parallel(sum_arg_42,ceil(0.5*n)-1,floor(n)-2)+py_sum_parallel(sum_arg_39,ceil(0.5*n),floor(n))+py_sum_parallel(sum_arg_37,ceil(0.5*n)-1,floor(n)-2)+is_seq(0,n-2)*is_integer(n-2)*py_sum_parallel(sum_arg_35,0,n-2)+py_sum_parallel(sum_arg_34,ceil(0.5*n),floor(n))/4-py_sum_parallel(sum_arg_32,ceil(0.5*n),floor(n))/2+py_sum_parallel(sum_arg_29,ceil(0.5*n),floor(n))-py_sum_parallel(sum_arg_27,ceil(0.5*n),floor(n))-py_sum_parallel(sum_arg_24,ceil(0.5*n),floor(n))))/(B_alpha_coef[0]*B_denom_coef_c[0]**3*n)))\
        +(-py_sum_parallel(sum_arg_9,ceil(0.5*n),floor(n))/2)\
        +(-py_sum_parallel(sum_arg_7,ceil(0.5*n),floor(n)))\
        +(py_sum_parallel(sum_arg_3,ceil(0.5*n),floor(n)))\
        +(py_sum_parallel(sum_arg_23,ceil(0.5*n),floor(n)))\
        +(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_20,0,n))\
        +(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_18,0,n))\
        +(-py_sum_parallel(sum_arg_15,ceil(0.5*n),floor(n)))\
        +(-is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_13,0,n))\
        +(py_sum_parallel(sum_arg_12,ceil(0.5*n),floor(n))/2)
    return(out)
