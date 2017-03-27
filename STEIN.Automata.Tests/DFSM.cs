using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using STEIN.Automata;
using System.Collections.Generic;

namespace STEIN.Automata.Tests
{
    [TestClass]
    public class DFSM
    {
        [TestMethod]
        public void DFSMObject()
        {
            var q0 = "q0";
            var q1 = 10;
            var q2 = new object();

            var trans = 420.69f;

            var Q = new List<object> { q0, q1, q2 };
            var Sigma = new List<float> { trans };
            var Delta = new List<Transition<object, float>>{
    new Transition<object, float>(q0, trans ,q1),
    new Transition<object, float>(q1, trans, q2),
    new Transition<object, float>(q2, trans, q1)
};
            var F = new List<object> { q0, q2 };
            var dFSM = new DeterministicFiniteStateMachine<object, float>(Q, Sigma, Delta, q0, F);

            Assert.IsFalse(dFSM.Accepts(new List<float> { trans }));
            Assert.IsTrue(dFSM.Accepts(new List<float> { }));
            Assert.IsFalse(dFSM.Accepts(new List<float> { trans }));
            Assert.IsTrue(dFSM.Accepts(new List<float> { trans, trans }));
            Assert.IsFalse(dFSM.Accepts(new List<float> { trans, trans, trans }));

        }
    }
}
