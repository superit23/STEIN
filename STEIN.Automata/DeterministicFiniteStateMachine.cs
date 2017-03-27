using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace STEIN.Automata
{
    /// <summary>
    /// 
    /// Adapted from http://bezensek.com/blog/2015/05/08/deterministic-finite-state-machine-minimization/
    /// </summary>
    /// <typeparam name="TState"></typeparam>
    /// <typeparam name="TSymbol"></typeparam>
    public class DeterministicFiniteStateMachine<TState, TSymbol>
    {
        private readonly List<TState> Q = new List<TState>();
        private readonly List<TSymbol> Sigma = new List<TSymbol>();
        private readonly List<Transition<TState, TSymbol>> Delta = new List<Transition<TState, TSymbol>>();
        private TState Q0;
        private readonly List<TState> F = new List<TState>();

        public TState CurrentState
        { get; private set; }

        public DeterministicFiniteStateMachine(IEnumerable<TState> q, IEnumerable<TSymbol> sigma,
                                IEnumerable<Transition<TState, TSymbol>> delta, TState q0,
                                IEnumerable<TState> f)
        {
            Q = q.ToList();
            Sigma = sigma.ToList();
            AddTransitions(delta);
            AddInitialState(q0);
            AddFinalStates(f);

            CurrentState = Q0;
        }

        private void AddInitialState(TState q0)
        {
            if (Q.Contains(q0))
            {
                Q0 = q0;
            }
        }

        private void AddFinalStates(IEnumerable<TState> finalStates)
        {
            foreach (var finalState in finalStates.Where(
                       finalState => Q.Contains(finalState)))
            {
                F.Add(finalState);
            }
        }

        private void AddTransitions(IEnumerable<Transition<TState, TSymbol>> transitions)
        {
            foreach (var transition in transitions.Where(ValidTransition))
            {
                Delta.Add(transition);
            }
        }

        private bool ValidTransition(Transition<TState, TSymbol> transition)
        {
            return Q.Contains(transition.StartState) &&
                    Q.Contains(transition.EndState) &&
                    Sigma.Contains(transition.Symbol) &&
                    !TransitionAlreadyDefined(transition);
        }

        private bool TransitionAlreadyDefined(Transition<TState, TSymbol> transition)
        {
            return Delta.Any(t => t.StartState.Equals(transition.StartState) &&
                                  t.Symbol.Equals(transition.Symbol));
        }

        /// <summary>
        /// Determines whether a list of symbols is within its language.
        /// </summary>
        /// <param name="input">Chain of transitions to take.</param>
        /// <returns>A <see cref="bool"/> indicating whether it reached a final state.</returns>
        public bool Accepts(IEnumerable<TSymbol> input)
        {
            var currentState = Q0;
            foreach (var symbol in input)
            {
                var transition = Delta.Find(t => t.StartState.Equals(currentState) &&
                                                 t.Symbol.Equals(symbol));
                if (transition == null)
                {
                    return false;
                }
                currentState = transition.EndState;
            }

            return F.Contains(currentState);
        }

        //public void Move(IEnumerable<TSymbol> input)
        //{
        //    foreach (var symbol in input)
        //    {
        //        var transition = Delta.Find(t => t.StartState.Equals(CurrentState) &&
        //                                         t.Symbol.Equals(symbol));
        //        if (transition == null)
        //        {
        //            throw new InvalidOperationException("No transitions for current state and symbol.");
        //        }
        //        CurrentState = transition.EndState;
        //    }
        //    if (!F.Contains(CurrentState))
        //    {
        //        throw new InvalidOperationException("Stopped in a state that is not final.");
        //    }

        //}

        /// <summary>
        /// Makes a transition using the provided symbol.
        /// </summary>
        /// <param name="input">The symbol to transition on.</param>
        /// <returns>A <see cref="bool"/> indicating whether it reached a final state.</returns>
        public bool Move(TSymbol input)
        {
            var transition = Delta.Find(t => t.StartState.Equals(CurrentState) &&
                                             t.Symbol.Equals(input));
            if (transition == null)
            {
                throw new InvalidOperationException("No transitions for current state and symbol.");
            }

            CurrentState = transition.EndState;

            return F.Contains(CurrentState);

        }

        public void Reset()
        {
            CurrentState = Q0;
        }
    }
}
