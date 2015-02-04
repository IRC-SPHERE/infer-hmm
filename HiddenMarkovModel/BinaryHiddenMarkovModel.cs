﻿//
// TemporalSegmentations.cs
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
    public class BinaryHiddenMarkovModel
    {
        /// <summary>
        /// The emission data
        /// </summary>
        private bool[] emitData;

        /// <summary>
        /// The k.
        /// </summary>
        private readonly Range k;

        /// <summary>
        /// The t.
        /// </summary>
        private readonly Range T;

        /// <summary>
        /// The states.
        /// </summary>
        private readonly VariableArray<int> states;

        /// <summary>
        /// The emissions.
        /// </summary>
        private readonly VariableArray<bool> emissions;

        /// <summary>
        /// The prob init.
        /// </summary>
        private readonly Variable<Vector> probInit;

        /// <summary>
        /// The cpt trans.
        /// </summary>
        private readonly VariableArray<Vector> cptTrans;

        /// <summary>
        /// The emit.
        /// </summary>
        private readonly VariableArray<double> emit;

        /// <summary>
        /// The prob init prior.
        /// </summary>
        private readonly Variable<Dirichlet> probInitPrior;

        /// <summary>
        /// The cpt trans prior.
        /// </summary>
        private readonly VariableArray<Dirichlet> cptTransPrior;

        /// <summary>
        /// The emit prior.
        /// </summary>
        private readonly VariableArray<Beta> emitPrior;
        
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
        /// The emit posterior.
        /// </summary>
        internal Beta[] emitPosterior;

        /// <summary>
        /// The states posterior.
        /// </summary>
        internal Discrete[] statesPosterior;

        /// <summary>
        /// The model evidence posterior.
        /// </summary>
        internal Bernoulli modelEvidencePosterior;

        /// <summary>
        /// Initializes a new instance of the <see cref="BinaryHiddenMarkovModel" /> class.
        /// </summary>
        /// <param name="ChainLength">Chain length.</param>
        /// <param name="NumStates">Number states.</param>
        public BinaryHiddenMarkovModel(int ChainLength, int NumStates)
        {
            this.modelEvidence = Variable.Bernoulli(0.5).Named("evidence");
            using (Variable.If(this.modelEvidence))
            {
                this.k = new Range(NumStates).Named("K");
                T = new Range(ChainLength).Named("T");

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
                this.emitPrior = Variable.Array<Beta>(this.k).Named("EmitPrior");
                this.emit = Variable.Array<double>(this.k).Named("Emit");
                this.emit[this.k] = Variable<double>.Random(this.emitPrior[this.k]);
                
                // Emit.SetValueRangeK);
                
                // Define the primary variables
                Variable<int> zeroState = Variable.Discrete(this.probInit).Named("z0");
                this.states = Variable.Array<int>(T);
                this.emissions = Variable.Array<bool>(T);

                // for block over length of chain
                using (var block = Variable.ForEach(T))
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
                        this.emissions[t] = Variable.Bernoulli(this.emit[this.states[t]]);
                    }   
                }
            }

            DefineInferenceEngine();
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
        public void ObserveData(bool[] data)
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
            this.emitPosterior = this.engine.Infer<Beta[]>(this.emit);
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
                bool emit = this.emissions[i].ObservedValue;
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
            this.emitPrior.ObservedValue = Util.ArrayInit(this.k.SizeAsInt, k => new Beta(1, 1)).ToArray(); // Gaussian.FromMeanAndVariance(1000, 1000000000)).ToArray();
        }

        /// <summary>
        /// Sets the priors.
        /// </summary>
        /// <param name="ProbInitPriorParamObs">Prob init prior parameter obs.</param>
        /// <param name="CPTTransPriorObs">CPT trans prior obs.</param>
        /// <param name="EmitPriorObs">Emit prior obs.</param>
        public void SetPriors(Dirichlet ProbInitPriorParamObs, Dirichlet[] CPTTransPriorObs, Beta[] EmitPriorObs)
        {
            this.probInitPrior.ObservedValue = ProbInitPriorParamObs;
            this.cptTransPrior.ObservedValue = CPTTransPriorObs;
            this.emitPrior.ObservedValue = EmitPriorObs;
        }

        /// <summary>
        /// Sets the parameters.
        /// </summary>
        /// <param name="init">Init.</param>
        /// <param name="trans">Trans.</param>
        /// <param name="emit">Emit.</param>
        public void SetParameters(double[] init, double[][] trans, double[] emit)
        {
            // fix parameters
            this.probInit.ObservedValue = Vector.FromArray(init);
            var v = new Vector[trans.Length];
            for (int i = 0; i < trans.Length; i++)
            {
                v[i] = Vector.FromArray(trans[i]);
            }
            this.cptTrans.ObservedValue = v;
            this.emit.ObservedValue = emit;
        }

        /// <summary>
        /// Sets the parameters to MAP estimates.
        /// </summary>
        public void SetParametersToMAPEstimates()
        {
            var trans = new Vector[this.k.SizeAsInt];
            var emit = new double[this.k.SizeAsInt];
            for (int i = 0; i < this.k.SizeAsInt; i++)
            {
                trans[i] = this.cptTransPosterior[i].PseudoCount;
                emit[i] = this.emitPosterior[i].GetMean();
            }
            this.probInit.ObservedValue = this.probInitPosterior.PseudoCount;
            this.cptTrans.ObservedValue = trans;
            this.emit.ObservedValue = emit;
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
            // emit mean
            for (int i = 0; i < this.k.SizeAsInt; i++)
            {
                returnString += this.emitPrior.ObservedValue[i].GetMean() + " ";
            }
            returnString += "\n";
            // emit var
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

            // output emit
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
