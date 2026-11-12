import json,numpy as np,scipy.sparse as sp,osqp
from relu_solver import ReLUQP
def main():
    rng=np.random.default_rng(7301);n=24;m=48
    M=rng.normal(size=(n,n));H=M.T@M+np.eye(n);g=rng.normal(size=n)*4
    A=rng.normal(size=(m,n));l=-np.ones(m)*.3;u=np.ones(m)*.3
    solver=ReLUQP(H,A,rho=1.);x=solver.solve(g,l,u,iterations=10000)
    ref=osqp.OSQP();ref.setup(P=sp.csc_matrix(H),q=g,A=sp.csc_matrix(A),l=l,u=u,verbose=False,eps_abs=1e-9,eps_rel=1e-9,max_iter=20000,polishing=True)
    result=ref.solve();delta=np.max(np.abs(x-result.x))
    assert result.info.status_val==1 and delta<1e-5,(delta,solver.info)
    assert solver.info['primal']<1e-6 and solver.info['dual']<1e-5,solver.info
    # Changing only g/l/u and a real nonzero warm start must hit the same answer.
    g=g+.1;x=solver.solve(g,l,u,iterations=10000,xwarm=x)
    ref.update(q=g);result=ref.solve();delta2=np.max(np.abs(x-result.x))
    assert delta2<1e-5,delta2
    print(json.dumps(dict(test='ReLU-QP versus OSQP active constraints and warm start',max_error=delta,updated_error=delta2,**solver.info)))
if __name__=='__main__':main()
