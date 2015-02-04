//
// DiscreteHiddenMarkovModel.cs
//
// Author:
//       Tom Diethe <tom.diethe@bristol.ac.uk>
//
// Copyright (c) 2015 University of Bristol
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

namespace HiddenMarkovModel
{
    using System;
    using System.Linq;

    using MicrosoftResearch.Infer;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;
    using MicrosoftResearch.Infer.Models;
    using MicrosoftResearch.Infer.Utils;

    /// <summary>
    /// Hidden markov model.
    /// </summary>
    public class DiscreteHiddenMarkovModel
    {
        /// <summary>
        /// The emission data
        /// </summary>
        private int[] emitData;

        /// <summary>
        /// The k.
        /// </summary>
        private readonly Range k;

        /// <summary>
        /// The t.
        /// </summary>
        private readonly Range T;

        /// <summary>
        /// The emmission range.
        /// </summary>
        private readonly Range emmissionRange;

        /// <summary>
        /// The states.
        /// </summary>
        private readonly VariableArray<int> states;

        /// <summary>
        /// The emissions.
        /// </summary>
        private readonly VariableArray<int> emissions;

        /// <summary>
        /// The prob init.
        /// </summary>
        private readonly Variable<Vector> probInit;

        /// <summary>
        /// The cpt trans.
        /// </summary>
        private readonly VariableArray<Vector> cptTrans;

        /// <summary>
        /// The emitObs.
        /// </summary>
        private readonly VariableArray<Vector> emit;

        /// <summary>
        /// The prob init prior.
        /// </summary>
        private readonly Variable<Dirichlet> probInitPrior;

        /// <summary>
        /// The cpt trans prior.
        /// </summary>
        private readonly VariableArray<Dirichlet> cptTransPrior;

        /// <summary>
        /// The emitObs prior.
        /// </summary>
        private readonly VariableArray<Dirichlet> emitPrior;
        
        // Set up model evidence (likelihood of model given data)
        private readonly Variable<bool> modelEvidence;

        // Inference engine
        private InferenceEngine engine;

        // Set up posteriors
        private Dirichlet probInitPosterior;

        /// <summary>
        /// The cpt trans posterior.
        /// </summary>
        private Dirichlet[] cptTransPosterior;

        /// <summary>
        /// The emitObs posterior.
        /// </summary>
        internal Dirichlet[] emitPosterior;

        /// <summary>
        /// The states posterior.
        /// </summary>
        internal Discrete[] statesPosterior;

        /// <summary>
        /// The model evidence posterior.
        /// </summary>
        internal Bernoulli modelEvidencePosterior;

        /// <summary>
        /// Initializes a new instance of the <see cref="DiscreteHiddenMarkovModel" /> class.
        /// </summary>
        /// <param name="chainLength">Chain length.</param>
        /// <param name="numStates">Number states.</param>
        public DiscreteHiddenMarkovModel(int chainLength, int numStates, int emmissionValues)
        {
            this.modelEvidence = Variable.Bernoulli(0.5).Named("evidence");
            using (Variable.If(this.modelEvidence))
            {
                this.k = new Range(numStates).Named("K");
                this.T = new Range(chainLength).Named("T");
                this.emmissionRange = new Range(emmissionValues).Named("emmissionRange");

                // Init
                this.probInitPrior = Variable.New<Dirichlet>().Named("ProbInitPrior");
                this.probInit = Variable<Vector>.Random(this.probInitPrior).Named("ProbInit");
                this.probInit.SetValueRange(this.k);
                
                // Trans probability table based on init
                this.cptTransPrior = Variable.Array<Dirichlet>(this.k).Named("CPTTransPrior");
                this.cptTrans = Variable.Array<Vector>(this.k).Named("CPTTrans");
                this.cptTrans[this.k] = Variable<Vector>.Random(this.cptTransPrior[this.k]);
                this.cptTrans.SetValueRange(this.k);
                
                // Emit prior
                this.emitPrior = Variable.Array<Dirichlet>(this.k).Named("EmitPrior");
                this.emit = Variable.Array<Vector>(this.k).Named("Emit");
                this.emit[this.k] = Variable<Vector>.Random(this.emitPrior[this.k]);
                this.emit.SetValueRange(this.emmissionRange);

                // Emit.SetValueRangeK);
                
                // Define the primary variables
                Variable<int> zeroState = Variable.Discrete(this.probInit).Named("z0");
                this.states = Variable.Array<int>(this.T);
                this.emissions = Variable.Array<int>(this.T);

                // for block over length of chain
                using (var block = Variable.ForEach(this.T))
                {
                    var t = block.Index;
                    var previousState = this.states[t - 1];

                    // initial distribution
                    using (Variable.If((t == 0).Named("Initial")))
                    {
                        using (Variable.Switch(zeroState))
                        {
                            this.states[T] = Variable.Discrete(this.cptTrans[zeroState]);
                        }
                    }

                    // transition distributions
                    using (Variable.If((t > 0).Named("Transition")))
                    {
                        using (Variable.Switch(previousState))
                        {
                            this.states[t] = Variable.Discrete(this.cptTrans[previousState]);
                        }
                    }                         

                    // emission distribution
                    using (Variable.Switch(this.states[t]))
                    {
                        this.emissions[t] = Variable.Discrete(this.emit[this.states[t]]);
                    }   
                }
            }

            this.DefineInferenceEngine();
        }

        /// <summary>
        /// Defines the inference engine.
        /// </summary>
        public void DefineInferenceEngine()
        {
            // Set up inference engine
            this.engine = new InferenceEngine(new ExpectationPropagation())
                              {
                                  ShowFactorGraph = true,
                                  ShowWarnings = true,
                                  ShowProgress = true,
                                  NumberOfIterations = 15,
                                  ShowTimings = true,
                                  ShowSchedule = false
                              };
            this.engine.Compiler.WriteSourceFiles = true;
        }

        /// <summary>
        /// Initialises the states randomly.
        /// </summary>
        public void InitialiseStatesRandomly()
        {
            var zinit = Variable<Discrete>.Array(T);
            zinit.ObservedValue = Util.ArrayInit(T.SizeAsInt, t => Discrete.PointMass(Rand.Int(this.k.SizeAsInt), this.k.SizeAsInt));
            this.states[T].InitialiseTo(zinit[T]);
        }

        /// <summary>
        /// Observes the data.
        /// </summary>
        /// <param name="data">Emit data.</param>
        public void ObserveData(int[] data)
        {
            // Save data as instance variable
            this.emitData = data;
            
            // Observe it
            this.emissions.ObservedValue = data;
        }

        /// <summary>
        /// Infers the posteriors.
        /// </summary>
        public void InferPosteriors()
        {
            // for monitoring convergence
            //for (int i = 1; i <= 35; i++ )
            //{
            //    Engine.NumberOfIterations = i;
            //    Console.WriteLine(CPTTransPosterior[0]);
            //}

            //infer posteriors
            this.cptTransPosterior = this.engine.Infer<Dirichlet[]>(this.cptTrans);
            this.probInitPosterior = this.engine.Infer<Dirichlet>(this.probInit);
            this.emitPosterior = this.engine.Infer<Dirichlet[]>(this.emit);
            this.statesPosterior = this.engine.Infer<Discrete[]>(this.states);
            this.modelEvidencePosterior = this.engine.Infer<Bernoulli>(this.modelEvidence);
        }

        /// <summary>
        /// Resets the inference.
        /// </summary>
        public void ResetInference()
        {
            // reset observations
            for (int i = 0; i < T.SizeAsInt; i++)
            {
                var emit = this.emissions[i].ObservedValue;
                this.emissions[i].ClearObservedValue();
                this.emissions[i].ObservedValue = emit;
            }
        }

        /// <summary>
        /// Sets the uninformed priors.
        /// </summary>
        public void SetUninformedPriors()
        {
            this.probInitPrior.ObservedValue = Dirichlet.Uniform(this.k.SizeAsInt);
            this.cptTransPrior.ObservedValue = Util.ArrayInit(this.k.SizeAsInt, k => Dirichlet.Uniform(this.k.SizeAsInt)).ToArray();
            this.emitPrior.ObservedValue = Util.ArrayInit(this.k.SizeAsInt, k => Dirichlet.Uniform(this.emmissionRange.SizeAsInt)).ToArray(); // Gaussian.FromMeanAndVariance(1000, 1000000000)).ToArray();
        }

        /// <summary>
        /// Sets the priors.
        /// </summary>
        /// <param name="probInitPriorParamObs">Prob init prior parameter obs.</param>
        /// <param name="cptTransPriorObs">CPT trans prior obs.</param>
        /// <param name="emitPriorObs">Emit prior obs.</param>
        public void SetPriors(Dirichlet probInitPriorParamObs, Dirichlet[] cptTransPriorObs, Dirichlet[] emitPriorObs)
        {
            this.probInitPrior.ObservedValue = probInitPriorParamObs;
            this.cptTransPrior.ObservedValue = cptTransPriorObs;
            this.emitPrior.ObservedValue = emitPriorObs;
        }

        /// <summary>
        /// Sets the parameters.
        /// </summary>
        /// <param name="init">Init.</param>
        /// <param name="trans">Trans.</param>
        /// <param name="emitObs">Emit.</param>
        public void SetParameters(double[] init, double[][] trans, Vector[] emitObs)
        {
            // fix parameters
            this.probInit.ObservedValue = Vector.FromArray(init);
            var v = new Vector[trans.Length];
            for (int i = 0; i < trans.Length; i++)
            {
                v[i] = Vector.FromArray(trans[i]);
            }

            this.cptTrans.ObservedValue = v;
            this.emit.ObservedValue = emitObs;
        }

        /// <summary>
        /// Sets the parameters to MAP estimates.
        /// </summary>
        public void SetParametersToMAPEstimates()
        {
            var trans = new Vector[this.k.SizeAsInt];
            var emitObs = new Vector[this.k.SizeAsInt];
            for (int i = 0; i < this.k.SizeAsInt; i++)
            {
                trans[i] = this.cptTransPosterior[i].PseudoCount;
                emitObs[i] = this.emitPosterior[i].GetMean();
            }
            this.probInit.ObservedValue = this.probInitPosterior.PseudoCount;
            this.cptTrans.ObservedValue = trans;
            this.emit.ObservedValue = emitObs;
        }

        /// <summary>
        /// Prints the prior.
        /// </summary>
        public void PrintPrior()
        {
            Console.WriteLine(this.probInitPrior.ObservedValue);
            for (int i = 0; i < this.k.SizeAsInt; i++)
            {
                Console.WriteLine("[" + i + "]" + this.cptTransPrior.ObservedValue[i]);
            }
            for (int i = 0; i < this.k.SizeAsInt; i++)
            {
                Console.WriteLine("[" + i + "]" + this.emitPrior.ObservedValue[i]);
            }
        }

        /// <summary>
        /// Prints the parameters.
        /// </summary>
        public void PrintParameters()
        {
            Console.WriteLine(this.probInit.ObservedValue);
            for (int i = 0; i < this.k.SizeAsInt; i++)
            {
                Console.WriteLine("[" + i + "]" + this.cptTrans.ObservedValue[i]);
            }
            for (int i = 0; i < this.k.SizeAsInt; i++)
            {
                Console.WriteLine("[" + i + "]" + this.emit.ObservedValue[i]);
            }
        }

        /// <summary>
        /// Prints the posteriors.
        /// </summary>
        public void PrintPosteriors()
        {
            Console.WriteLine(this.probInitPosterior);
            for (int i = 0; i < this.k.SizeAsInt; i++)
            {
                Console.WriteLine("[" + i + "]" + this.cptTransPosterior[i]);
            }
            for (int i = 0; i < this.k.SizeAsInt; i++)
            {
                Console.WriteLine("[" + i + "]" + this.emitPosterior[i]);
            }
        }

        /// <summary>
        /// Hyperparameterses to string.
        /// </summary>
        /// <returns>The to string.</returns>
        public string HyperparametersToString()
        {
            string returnString = string.Empty;

            // init
            returnString += this.probInitPrior.ObservedValue.PseudoCount + "\n";
            // trans
            for (int i = 0; i < this.k.SizeAsInt; i++)
            {
                returnString += this.cptTransPrior.ObservedValue[i].PseudoCount + "\n";
            }
            // emitObs mean
            for (int i = 0; i < this.k.SizeAsInt; i++)
            {
                returnString += this.emitPrior.ObservedValue[i].GetMean() + " ";
            }
            returnString += "\n";
            // emitObs var
            for (int i = 0; i < this.k.SizeAsInt; i++)
            {
                returnString += this.emitPrior.ObservedValue[i].GetVariance() + " ";
            }
            returnString += "\n";
            
            return returnString;
        }

        /// <summary>
        /// Prints the states.
        /// </summary>
        public void PrintStates()
        {
            string output = "state, power" + "\n";
            for (int i = 0; i < T.SizeAsInt; i++)
            {
                output += this.engine.Infer<Discrete>(this.states[i]).GetMode() + ", " + this.emitData[i] + "\n";
            }

            Console.WriteLine(output);

            output = "state, power" + "\n";
            for (int i = 0; i < T.SizeAsInt; i++)
            {
                output += this.engine.Infer<Discrete>(this.states[i]) + ", " + this.emitData[i] + "\n";
            }

            Console.WriteLine(output);
        }

        /// <summary>
        /// Returns a <see cref="System.String"/> that represents the current <see cref="BinaryHiddenMarkovModel"/>.
        /// </summary>
        /// <returns>A <see cref="System.String"/> that represents the current <see cref="BinaryHiddenMarkovModel"/>.</returns>
        public override string ToString()
        {
            string output = string.Empty;
            const bool PrintInit = true;
            const bool PrintTrans = true;
            const bool PrintEmit = true;
            const bool PrintStates = true;

            // output init
            if (PrintInit)
            {
                output += "ProbInitPosterior" + this.probInitPosterior + "\n";
            }

            // output trans
            if (PrintTrans)
            {
                for (int i = 0; i < this.k.SizeAsInt; i++)
                {
                    output += "CPTTransPosterior[" + i + "]" + this.cptTransPosterior[i] + "\n";
                }
            }

            // output emitObs
            if (PrintEmit)
            {
                for (int i = 0; i < this.k.SizeAsInt; i++)
                {
                    output += "Emit Posterior[" + i + "] " + this.emitPosterior[i] + "\n";
                }
            }

            // output states
            if (PrintStates)
            {
                output += "state, power" + "\n";
                for (int i = 0; i < T.SizeAsInt; i++)
                {
                    Console.WriteLine(this.statesPosterior[i]);
                    output += this.statesPosterior[i].GetMode() + ", " + this.emissions[i].ObservedValue + "\n";
                }
            }

            return output;
        }
    }
}
