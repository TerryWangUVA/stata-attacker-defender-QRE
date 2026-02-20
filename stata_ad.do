* ---------- Parameters ----------
local T = 500
local gamma = 0.01          // step size toward logit BR (small)
local lamD  = 10           // defender precision (higher = more "rational")
local lamA  = 10           // attacker precision
local sig   = 0.00          // optional extra noise; set >0 if you want

clear
set more off
set seed 42
set obs `T'
gen t = _n

* State: mixed strategies
gen p = .   // Pr(High)
gen q = .   // Pr(Attack)

* Optional realized actions
gen aD = .
gen aA = .

* Initialize away from equilibrium
replace p = 0.85 in 1
replace q = 0.15 in 1

forvalues i = 1/`=`T'-1' {

    * Optional: sample realized actions from current mixed strategies
    replace aD = (runiform() < p[`i']) in `i'    // 1=High, 0=Low
    replace aA = (runiform() < q[`i']) in `i'    // 1=Attack, 0=Wait

    * ---- Compute expected utilities under beliefs (p_i, q_i) ----
	
**# asym
	
* ---- Compute expected utilities under beliefs (p_i, q_i) ----
* Game with (p*, q*) = (1/4, 3/4)
* Payoffs (Def, Att):
* High,Wait   = (0, 0)
* High,Attack = (3,-3)
* Low,Wait    = (3,-3)
* Low,Attack  = (2,-2)

* Defender EU conditional on q = Pr(Attack)
scalar EU_D_H = 3*(q[`i'])          // = 3q
scalar EU_D_L = 3 - (q[`i'])        // = 3 - q

* Attacker EU conditional on p = Pr(High)
scalar EU_A_A = -2 - (p[`i'])       // = -2 - p
scalar EU_A_W = -3 + 3*(p[`i'])     // = -3 + 3p


**# sym
	
	/*
	* Defender EU conditional on q = Pr(Attack)
	scalar EU_D_H = 4*(1 - q[`i']) + (-2)*q[`i']      // = 4 - 6q
	scalar EU_D_L = (-2)*(1 - q[`i']) + 4*q[`i']      // = -2 + 6q

	* Attacker EU conditional on p = Pr(High)
	* Attacker's action is Attack vs Wait
	scalar EU_A_A = 4*p[`i'] + (-2)*(1 - p[`i'])      // = -2 + 6p
	scalar EU_A_W = (-2)*p[`i'] + 4*(1 - p[`i'])      // = 4 - 6p
	*/
	
    * ---- Logit (Quantal) best responses ----
    * Use logit on payoff DIFFERENCE to avoid overflow:
    * P(High)=1/(1+exp(-lam*(EU_H - EU_L)))
    scalar p_br = 1/(1 + exp(-`lamD'*(EU_D_H - EU_D_L)))
    scalar q_br = 1/(1 + exp(-`lamA'*(EU_A_A - EU_A_W)))   // prob(Attack)
	
    * ---- Stochastic approximation update ----
    scalar pnext = (1-`gamma')*p[`i'] + `gamma'*p_br + `sig'*rnormal()
    scalar qnext = (1-`gamma')*q[`i'] + `gamma'*q_br + `sig'*rnormal()

    * clip to [0,1]
    if (pnext < 0) scalar pnext = 0
    if (pnext > 1) scalar pnext = 1
    if (qnext < 0) scalar qnext = 0
    if (qnext > 1) scalar qnext = 1

    replace p = pnext in `=`i'+1'
    replace q = qnext in `=`i'+1'
}


local pstar = 0.25
local qstar = 0.75

twoway ///
    (line p t, sort) ///
    (line q t, sort) ///
    , ///
    yline(`pstar' `qstar', lpattern(dash)) ///
    yscale(range(0 1)) ylabel(0(0.1)1) ///
    title("QRE process: p_t (High Alert) and q_t (Attack)") ///
    legend(order(1 "p_t = Pr(High Alert)" 2 "q_t = Pr(Attack)" ///
                 3 "p* = 0.25" 4 "q* = 0.75") pos(6) col(4))
