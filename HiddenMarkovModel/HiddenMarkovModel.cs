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
    public class HiddenMarkovModel
    {
        /// <summary>
        /// The emit data.
        /// </summary>
        private double[] emitData;

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
        private readonly VariableArray<double> emissions;

        /// <summary>
        /// The prob init.
        /// </summary>
        private readonly Variable<Vector> probInit;

        /// <summary>
        /// The cpt trans.
        /// </summary>
        private readonly VariableArray<Vector> cptTrans;

        /// <summary>
        /// The emit mean.
        /// </summary>
        private readonly VariableArray<double> emitMean;

        /// <summary>
        /// The emit prec.
        /// </summary>
        private readonly VariableArray<double> emitPrec;

        /// <summary>
        /// The prob init prior.
        /// </summary>
        private readonly Variable<Dirichlet> probInitPrior;

        /// <summary>
        /// The cpt trans prior.
        /// </summary>
        private readonly VariableArray<Dirichlet> cptTransPrior;

        /// <summary>
        /// The emit mean prior.
        /// </summary>
        private readonly VariableArray<Gaussian> emitMeanPrior;

        /// <summary>
        /// The emit prec prior.
        /// </summary>
        private readonly VariableArray<Gamma> emitPrecPrior;

        // Set up model evidence (likelihood of model given data)
        private readonly Variable<bool> modelEvidence;

        /// <summary>
        /// The engine.
        /// </summary>
        private InferenceEngine engine;

        /// <summary>
        /// The prob init posterior.
        /// </summary>
        private Dirichlet probInitPosterior;

        /// <summary>
        /// The cpt trans posterior.
        /// </summary>
        private Dirichlet[] cptTransPosterior;

        /// <summary>
        /// The emit mean posterior.
        /// </summary>
        internal Gaussian[] emitMeanPosterior;

        /// <summary>
        /// The emit prec posterior.
        /// </summary>
        private Gamma[] emitPrecPosterior;

        /// <summary>
        /// The states posterior.
        /// </summary>
        internal Discrete[] statesPosterior;

        internal Bernoulli modelEvidencePosterior;

        /// <summary>
        /// Initializes a new instance of the <see cref="HiddenMarkovModel"/> class.
        /// </summary>
        /// <param name="chainLength">Chain length.</param>
        /// <param name="numStates">Number states.</param>
        public HiddenMarkovModel(int chainLength, int numStates)
        {
            this.modelEvidence = Variable.Bernoulli(0.5).Named("evidence");
            using (Variable.If(this.modelEvidence))
            {
                this.k = new Range(numStates).Named("K");
                this.T = new Range(chainLength).Named("T");
                
                // Init
                this.probInitPrior = Variable.New<Dirichlet>().Named("ProbInitPrior");
                this.probInit = Variable<Vector>.Random(this.probInitPrior).Named("ProbInit");
                this.probInit.SetValueRange(this.k);
                
                // Trans probability table based on init
                this.cptTransPrior = Variable.Array<Dirichlet>(this.k).Named("CPTTransPrior");
                this.cptTrans = Variable.Array<Vector>(this.k).Named("CPTTrans");
                this.cptTrans[this.k] = Variable<Vector>.Random(this.cptTransPrior[this.k]);
                this.cptTrans.SetValueRange(this.k);
                
                // Emit mean
                this.emitMeanPrior = Variable.Array<Gaussian>(this.k).Named("EmitMeanPrior");
                this.emitMean = Variable.Array<double>(this.k).Named("EmitMean");
                this.emitMean[this.k] = Variable<double>.Random(this.emitMeanPrior[this.k]);
                this.emitMean.SetValueRange(this.k);
                
                // Emit prec
                this.emitPrecPrior = Variable.Array<Gamma>(this.k).Named("EmitPrecPrior");
                this.emitPrec = Variable.Array<double>(this.k).Named("EmitPrec");
                this.emitPrec[this.k] = Variable<double>.Random(this.emitPrecPrior[this.k]);
                this.emitPrec.SetValueRange(this.k);

                // Define the primary variables
                Variable<int> zeroState = Variable.Discrete(this.probInit).Named("z0");
                this.states = Variable.Array<int>(T);
                this.emissions = Variable.Array<double>(T);

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
                        this.emissions[t] = Variable.GaussianFromMeanAndPrecision(this.emitMean[this.states[t]], this.emitPrec[this.states[t]]);
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
            VariableArray<Discrete> zinit = Variable<Discrete>.Array(T);
            zinit.ObservedValue = Util.ArrayInit(T.SizeAsInt, t => Discrete.PointMass(Rand.Int(this.k.SizeAsInt), this.k.SizeAsInt));
            this.states[T].InitialiseTo(zinit[T]);
        }

        /// <summary>
        /// Observes the data.
        /// </summary>
        /// <param name="data">Emit data.</param>
        public void ObserveData(double[] data)
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
            // for (int i = 1; i <= 35; i++ )
            // {
            //    Engine.NumberOfIterations = i;
            //    Console.WriteLine(CPTTransPosterior[0]);
            // }

            // infer posteriors
            this.cptTransPosterior = this.engine.Infer<Dirichlet[]>(this.cptTrans);
            this.probInitPosterior = this.engine.Infer<Dirichlet>(this.probInit);
            this.emitMeanPosterior = this.engine.Infer<Gaussian[]>(this.emitMean);
            this.emitPrecPosterior = this.engine.Infer<Gamma[]>(this.emitPrec);
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
                double emit = this.emissions[i].ObservedValue;
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
            this.cptTransPrior.ObservedValue = Util.ArrayInit(this.k.SizeAsInt, ia => Dirichlet.Uniform(this.k.SizeAsInt)).ToArray();
            this.emitMeanPrior.ObservedValue = Util.ArrayInit(this.k.SizeAsInt, ia => Gaussian.FromMeanAndVariance(1000, 1000000000)).ToArray();
            this.emitPrecPrior.ObservedValue = Util.ArrayInit(this.k.SizeAsInt, ia => Gamma.FromMeanAndVariance(0.1, 100)).ToArray();
        }

        /// <summary>
        /// Sets the priors.
        /// </summary>
        /// <param name="ProbInitPriorParamObs">Prob init prior parameter obs.</param>
        /// <param name="CPTTransPriorObs">CPT trans prior obs.</param>
        /// <param name="EmitMeanPriorObs">Emit mean prior obs.</param>
        /// <param name="EmitPrecPriorObs">Emit prec prior obs.</param>
        public void SetPriors(Dirichlet ProbInitPriorParamObs, Dirichlet[] CPTTransPriorObs, Gaussian[] EmitMeanPriorObs, Gamma[] EmitPrecPriorObs)
        {
            this.probInitPrior.ObservedValue = ProbInitPriorParamObs;
            this.cptTransPrior.ObservedValue = CPTTransPriorObs;
            this.emitMeanPrior.ObservedValue = EmitMeanPriorObs;
            this.emitPrecPrior.ObservedValue = EmitPrecPriorObs;
        }

        /// <summary>
        /// Sets the parameters.
        /// </summary>
        /// <param name="init">Init.</param>
        /// <param name="trans">Trans.</param>
        /// <param name="emitMeans">Emit means.</param>
        /// <param name="emitPrecs">Emit precs.</param>
        public void SetParameters(double[] init, double[][] trans, double[] emitMeans, double[] emitPrecs)
        {
            // fix parameters
            this.probInit.ObservedValue = Vector.FromArray(init);
            Vector[] v = new Vector[trans.Length];
            for (int i = 0; i < trans.Length; i++)
            {
                v[i] = Vector.FromArray(trans[i]);
            }

            this.cptTrans.ObservedValue = v;
            this.emitMean.ObservedValue = emitMeans;
            this.emitPrec.ObservedValue = emitPrecs;
        }

        /// <summary>
        /// Sets the parameters to MAP estimates.
        /// </summary>
        public void SetParametersToMapEstimates()
        {
            var trans = new Vector[this.k.SizeAsInt];
            var emitMean = new double[this.k.SizeAsInt];
            var emitPrec = new double[this.k.SizeAsInt];
            for (int i = 0; i < this.k.SizeAsInt; i++)
            {
                trans[i] = this.cptTransPosterior[i].PseudoCount;
                emitMean[i] = this.emitMeanPosterior[i].GetMean();
                emitPrec[i] = this.emitPrecPosterior[i].GetMean();
            }

            this.probInit.ObservedValue = this.probInitPosterior.PseudoCount;
            this.cptTrans.ObservedValue = trans;
            this.emitMean.ObservedValue = emitMean;
            this.emitPrec.ObservedValue = emitPrec;
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
                Console.WriteLine("[" + i + "]" + this.emitMeanPrior.ObservedValue[i]);
            }

            for (int i = 0; i < this.k.SizeAsInt; i++)
            {
                Console.WriteLine("[" + i + "]" + this.emitPrecPrior.ObservedValue[i]);
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
                Console.WriteLine("[" + i + "]" + this.emitMean.ObservedValue[i]);
            }
            
            for (int i = 0; i < this.k.SizeAsInt; i++)
            {
                Console.WriteLine("[" + i + "]" + this.emitPrec.ObservedValue[i]);
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
                Console.WriteLine("[" + i + "]" + this.emitMeanPosterior[i]);
            }
            
            for (int i = 0; i < this.k.SizeAsInt; i++)
            {
                Console.WriteLine("[" + i + "]" + this.emitPrecPosterior[i]);
            }
        }

        /// <summary>
        /// Hyper-parameters to string.
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

            // emit mean mean
            for (int i = 0; i < this.k.SizeAsInt; i++)
            {
                returnString += this.emitMeanPrior.ObservedValue[i].GetMean() + " ";
            }
            
            returnString += "\n";
            
            // emit mean var
            for (int i = 0; i < this.k.SizeAsInt; i++)
            {
                returnString += this.emitMeanPrior.ObservedValue[i].GetVariance() + " ";
            }

            returnString += "\n";
            
            // emit prec shape
            for (int i = 0; i < this.k.SizeAsInt; i++)
            {
                returnString += this.emitPrecPrior.ObservedValue[i].Shape + " ";
            }
            
            returnString += "\n";
            
            // emit prec shape
            for (int i = 0; i < this.k.SizeAsInt; i++)
            {
                returnString += this.emitPrecPrior.ObservedValue[i].GetScale() + " ";
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
            for (int i = 0; i < this.T.SizeAsInt; i++)
            {
                output += this.engine.Infer<Discrete>(this.states[i]).GetMode() + ", " + this.emitData[i] + "\n";
            }
            
            Console.WriteLine(output);

            output = "state, power" + "\n";
            for (int i = 0; i < this.T.SizeAsInt; i++)
            {
                output += this.engine.Infer<Discrete>(this.states[i]) + ", " + this.emitData[i] + "\n";
            }
            
            Console.WriteLine(output);
        }

        /// <summary>
        /// Returns a <see cref="System.String"/> that represents the current <see cref="HiddenMarkovModel"/>.
        /// </summary>
        /// <returns>A <see cref="System.String"/> that represents the current <see cref="HiddenMarkovModel"/>.</returns>
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
                    output += "Emit Mean Posterior[" + i + "] " + this.emitMeanPosterior[i] + "\n";
                    output += "Emit Prec Posterior[" + i + "] " + this.emitPrecPosterior[i] + "\n";
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