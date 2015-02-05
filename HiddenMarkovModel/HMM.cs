//
// HMM.cs
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

    using MicrosoftResearch.Infer;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;
    using MicrosoftResearch.Infer.Models;
    using MicrosoftResearch.Infer.Utils;

    /// <summary>
    /// Hidden markov model.
    /// </summary>
	public abstract class HMM<TEmit, TEmitMeanDist, TEmitMean, TEmitPrecDist, TEmitPrec> 
		where TEmitMeanDist : IDistribution<TEmitMean>, CanGetMean<TEmitMean>
		where TEmitPrecDist : IDistribution<TEmitPrec> , CanGetMean<TEmitPrec>
    {
        /// <summary>
        /// The emit data.
        /// </summary>
		private TEmit[] emitData;

        /// <summary>
        /// The hidden states.
        /// </summary>
        private Range k;

        /// <summary>
        /// The chain length.
        /// </summary>
		private Range chain;

		/// <summary>
		/// The emission.
		/// </summary>
		private Range emission;

        /// <summary>
        /// The states.
        /// </summary>
        private VariableArray<int> states;

        /// <summary>
        /// The emissions.
        /// </summary>
		private VariableArray<TEmit> emissions;

        /// <summary>
        /// The prob init.
        /// </summary>
        private Variable<Vector> probInit;

        /// <summary>
        /// The cpt trans.
        /// </summary>
        private VariableArray<Vector> cptTrans;

        /// <summary>
        /// The emit mean.
        /// </summary>
		private VariableArray<TEmitMean> emitMean;

        /// <summary>
        /// The emit prec.
        /// </summary>
		private VariableArray<TEmitPrec> emitPrec;

        /// <summary>
        /// The prob init prior.
        /// </summary>
        private Variable<Dirichlet> probInitPrior;

        /// <summary>
        /// The cpt trans prior.
        /// </summary>
        private VariableArray<Dirichlet> cptTransPrior;

        /// <summary>
        /// The emit mean prior.
        /// </summary>
		private VariableArray<TEmitMeanDist> emitMeanPrior;

        /// <summary>
        /// The emit prec prior.
        /// </summary>
		private VariableArray<TEmitPrecDist> emitPrecPrior;

        /// <summary>
        /// The model evidence.
        /// </summary>
        private Variable<bool> modelEvidence;

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
		internal TEmitMeanDist[] emitMeanPosterior;

        /// <summary>
        /// The emit prec posterior.
        /// </summary>
		private TEmitPrecDist[] emitPrecPosterior;

        /// <summary>
        /// The states posterior.
        /// </summary>
        internal Discrete[] statesPosterior;

		/// <summary>
		/// The model evidence posterior.
		/// </summary>
        internal Bernoulli modelEvidencePosterior;

        /// <summary>
        /// Creates the model.
        /// </summary>
        /// <param name="chainLength">Chain length.</param>
        /// <param name="numStates">Number states.</param>
        /// <param name="emissionDimension">Emission dimension.</param>
        /// <param name="includePrecision">If set to <c>true</c> include precision.</param>
		public virtual void CreateModel(int chainLength, int numStates, int emissionDimension = 1, bool includePrecision = true, bool showFactorGraph = true)
        {
            this.modelEvidence = Variable.Bernoulli(0.5).Named("evidence");
            using (Variable.If(this.modelEvidence))
            {
                this.k = new Range(numStates).Named("K");
                this.chain = new Range(chainLength).Named("chain");
				if (emissionDimension > 1)
				{
					this.emission = new Range(emissionDimension).Named("emission");
				}

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
                this.emitMeanPrior = Variable.Array<TEmitMeanDist>(this.k).Named("EmitMeanPrior");
				this.emitMean = Variable.Array<TEmitMean>(this.k).Named("EmitMean");
				this.emitMean[this.k] = Variable<TEmitMean>.Random(this.emitMeanPrior[this.k]);
				this.emitMean.SetValueRange(emissionDimension > 1 ? this.emission : this.k);
                
                // Emit prec
				if (includePrecision)
				{
					this.emitPrecPrior = Variable.Array<TEmitPrecDist>(this.k).Named("EmitPrecPrior");
					this.emitPrec = Variable.Array<TEmitPrec>(this.k).Named("EmitPrec");
					this.emitPrec[this.k] = Variable<TEmitPrec>.Random(this.emitPrecPrior[this.k]);
					this.emitPrec.SetValueRange(this.k);
				}

                // Define the primary variables
                Variable<int> zeroState = Variable.Discrete(this.probInit).Named("z0");
                this.states = Variable.Array<int>(chain);
                this.emissions = Variable.Array<TEmit>(chain);

                // for block over length of chain
                using (var block = Variable.ForEach(chain))
                {
                    var t = block.Index;
                    var previousState = this.states[t - 1];

                    // initial distribution
                    using (Variable.If((t == 0).Named("Initial")))
                    {
                        using (Variable.Switch(zeroState))
                        {
                            this.states[chain] = Variable.Discrete(this.cptTrans[zeroState]);
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
						this.emissions[t] = this.EmissionFunction(this.emitMean, this.emitPrec, this.states[t]);
                    }   
                }
            }

            DefineInferenceEngine(showFactorGraph);
        }

		/// <summary>
		/// The Emission function.
		/// </summary>
		/// <returns>The function.</returns>
		/// <param name="means">Means.</param>
		/// <param name="precs">Precs.</param>
		/// <param name="state">State.</param>
		public abstract Variable<TEmit> EmissionFunction(VariableArray<TEmitMean> means, VariableArray<TEmitPrec> precs, Variable<int> state);

        /// <summary>
        /// Defines the inference engine.
        /// </summary>
		public void DefineInferenceEngine(bool showFactorGraph)
        {
            // Set up inference engine
			this.engine = new InferenceEngine(new ExpectationPropagation())
			{
				ShowFactorGraph = showFactorGraph,
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
            VariableArray<Discrete> zinit = Variable<Discrete>.Array(chain);
            zinit.ObservedValue = Util.ArrayInit(chain.SizeAsInt, t => Discrete.PointMass(Rand.Int(this.k.SizeAsInt), this.k.SizeAsInt));
            this.states[chain].InitialiseTo(zinit[chain]);
        }

        /// <summary>
        /// Observes the data.
        /// </summary>
        /// <param name="data">Emit data.</param>
		public void ObserveData(TEmit[] data)
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
            // infer posteriors
            this.cptTransPosterior = this.engine.Infer<Dirichlet[]>(this.cptTrans);
            this.probInitPosterior = this.engine.Infer<Dirichlet>(this.probInit);
			this.emitMeanPosterior = this.engine.Infer<TEmitMeanDist[]>(this.emitMean);
			if (!ReferenceEquals(this.emitPrecPrior, null))
			{
				this.emitPrecPosterior = this.engine.Infer<TEmitPrecDist[]>(this.emitPrec);
			}

            this.statesPosterior = this.engine.Infer<Discrete[]>(this.states);
            this.modelEvidencePosterior = this.engine.Infer<Bernoulli>(this.modelEvidence);
        }

        /// <summary>
        /// Resets the inference.
        /// </summary>
        public void ResetInference()
        {
            // reset observations
            for (int i = 0; i < chain.SizeAsInt; i++)
            {
				var emit = this.emissions[i].ObservedValue;
                this.emissions[i].ClearObservedValue();
                this.emissions[i].ObservedValue = emit;
            }
        }

        /// <summary>
        /// Sets the priors.
        /// </summary>
        /// <param name="ProbInitPriorParamObs">Prob init prior parameter obs.</param>
        /// <param name="CPTTransPriorObs">CPT trans prior obs.</param>
        /// <param name="EmitMeanPriorObs">Emit mean prior obs.</param>
        /// <param name="EmitPrecPriorObs">Emit prec prior obs.</param>
		public void SetPriors(Dirichlet ProbInitPriorParamObs, Dirichlet[] CPTTransPriorObs, TEmitMeanDist[] EmitMeanPriorObs, TEmitPrecDist[] EmitPrecPriorObs)
        {
            this.probInitPrior.ObservedValue = ProbInitPriorParamObs;
            this.cptTransPrior.ObservedValue = CPTTransPriorObs;
            this.emitMeanPrior.ObservedValue = EmitMeanPriorObs;
			if (!ReferenceEquals(this.emitPrecPrior, null))
			{
				this.emitPrecPrior.ObservedValue = EmitPrecPriorObs;
			}
        }

        /// <summary>
        /// Sets the parameters.
        /// </summary>
        /// <param name="init">Init.</param>
        /// <param name="trans">Trans.</param>
        /// <param name="emitMeans">Emit means.</param>
        /// <param name="emitPrecs">Emit precs.</param>
		public void SetParameters(double[] init, double[][] trans, TEmitMean[] emitMeans, TEmitPrec[] emitPrecs)
        {
            // fix parameters
            this.probInit.ObservedValue = Vector.FromArray(init);
            var v = new Vector[trans.Length];
            for (int i = 0; i < trans.Length; i++)
            {
                v[i] = Vector.FromArray(trans[i]);
            }

            this.cptTrans.ObservedValue = v;
            this.emitMean.ObservedValue = emitMeans;
			if (!ReferenceEquals(this.emitPrecPrior, null))
			{
				this.emitPrec.ObservedValue = emitPrecs;
			}
        }

        /// <summary>
        /// Sets the parameters to MAP estimates.
        /// </summary>
        public void SetParametersToMapEstimates()
        {
            var trans = new Vector[this.k.SizeAsInt];
			var mean = new TEmitMean[this.k.SizeAsInt];
			var prec = new TEmitPrec[this.k.SizeAsInt];
            for (int i = 0; i < this.k.SizeAsInt; i++)
            {
                trans[i] = this.cptTransPosterior[i].PseudoCount;
                mean[i] = this.emitMeanPosterior[i].GetMean();
                prec[i] = this.emitPrecPosterior[i].GetMean();
            }

            this.probInit.ObservedValue = this.probInitPosterior.PseudoCount;
            this.cptTrans.ObservedValue = trans;
            this.emitMean.ObservedValue = mean;
			if (!ReferenceEquals(this.emitPrecPrior, null))
			{
				this.emitPrec.ObservedValue = prec;
			}
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

			if (ReferenceEquals(this.emitPrecPrior, null))
			{
				return;
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
            
			if (ReferenceEquals(this.emitPrec, null))
			{
				return;
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

			if (ReferenceEquals(this.emitPrecPrior, null))
			{
				return;
			}
            
            for (int i = 0; i < this.k.SizeAsInt; i++)
            {
                Console.WriteLine("[" + i + "]" + this.emitPrecPosterior[i]);
            }
        }

        /// <summary>
        /// Prints the states.
        /// </summary>
        public void PrintStates()
        {
            string output = "state, power" + "\n";
            for (int i = 0; i < this.chain.SizeAsInt; i++)
            {
                output += this.engine.Infer<Discrete>(this.states[i]).GetMode() + ", " + this.emitData[i] + "\n";
            }
            
            Console.WriteLine(output);

            output = "state, power" + "\n";
            for (int i = 0; i < this.chain.SizeAsInt; i++)
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
                for (int i = 0; i < chain.SizeAsInt; i++)
                {
                    Console.WriteLine(this.statesPosterior[i]);
                    output += this.statesPosterior[i].GetMode() + ", " + this.emissions[i].ObservedValue + "\n";
                }
            }

            return output;
        }
    }
}