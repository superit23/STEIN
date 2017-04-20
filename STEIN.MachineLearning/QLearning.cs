using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using STEIN.Automata;

namespace STEIN.MachineLearning
{
    public class QLearning<TState, TSymbol> : DeterministicFiniteStateMachine<TState, TSymbol>
    {
        /// <summary>
        /// Determines whether we should pick a random value.
        /// </summary>
        public float Epsilon
        { get; set; }


        /// <summary>
        /// Discount factor for states leading up to reward.
        /// </summary>
        public float Discount
        { get; set; }

        public Dictionary<Transition<TState, TSymbol>, float> StateRewards
        { get; set; }


        public QLearning(IEnumerable<TState> q, IEnumerable<TSymbol> sigma, IEnumerable<Transition<TState, TSymbol>> delta, TState q0, IEnumerable<TState> f, float epsilon = 0.1f, float discount = 0.2f) : base(q, sigma, delta, q0, f)
        {
            Epsilon = epsilon;
            StateRewards = new Dictionary<Transition<TState, TSymbol>, float>();
        }
    }
}
