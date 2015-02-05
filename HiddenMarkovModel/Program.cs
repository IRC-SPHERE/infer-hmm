//
// Program.cs
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
	using System.Collections.Generic;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>
    /// The Program.
    /// </summary>
    public class Program
    {
        /// <summary>
        /// The entry point of the program, where the program control starts and ends.
        /// </summary>
        public static void Main()
        {
			const int T = 100;
			const int K = 2;
			const int N = 5;
			const bool showFactorGraph = false;

			TestHMM<ContinousHMM, double, double, Gaussian, Gaussian, double, Gamma, double>(
				T,
				K,
				1,
				Gaussian.FromMeanAndPrecision, 
				() => Gaussian.FromMeanAndVariance(0, 1000), 
				() => Gamma.FromShapeAndScale(1000, 0.001),
				showFactorGraph);

			// TModel, TEmit, TEmitDist, TEmitMeanDist, TEmitMean, TEmitPrecDist, TEmitPrec
			TestHMM<MultivariateHMM, Vector, Vector, VectorGaussian, VectorGaussian, Vector, Wishart, PositiveDefiniteMatrix>(
				T,
				K,
				1,
				VectorGaussian.FromMeanAndPrecision, 
				() => VectorGaussian.FromMeanAndVariance(Vector.Zero(N), PositiveDefiniteMatrix.IdentityScaledBy(N, 1000)), 
				() => Wishart.FromShapeAndScale(N, PositiveDefiniteMatrix.IdentityScaledBy(N, 0.001)),
				showFactorGraph);

			TestHMM<BinaryHMM, bool, double, Bernoulli, Beta, double, Beta, double>(
				T,
				K,
				1,
				(m, p) => new Bernoulli(m),
				() => new Beta(1, 1),
				null,
				showFactorGraph);

			TestHMM<DiscreteHMM, int, double, Discrete, Dirichlet, Vector, Dirichlet, Vector>(
				T,
				K,
				N,
				(m, p) => new Discrete(m),
				() => Dirichlet.Uniform(N),
				null,
				showFactorGraph);

			// TestBinaryHiddenMarkovModel();
            // TestDiscreteHiddenMarkovModel();
			// TestMultivariateHMM();
        }

		public static void TestHMM<TModel, TEmit, TEmitPosterior, TEmitDist, TEmitMeanDist, TEmitMean, TEmitPrecDist, TEmitPrec>(
			int chainLength,
			int hiddenStates,
			int emissionDimension,
			Func<TEmitMean, TEmitPrec, TEmitDist> emitPriorFunc, 
			Func<TEmitMeanDist> emitPriorMeanFunc,
			Func<TEmitPrecDist> emitPriorPrecFunc,
			bool showFactorGraph
		)
				where TModel : HMM<TEmit, TEmitMeanDist, TEmitMean, TEmitPrecDist, TEmitPrec>, new()
				where TEmitDist : IDistribution<TEmit>, CanGetMean<TEmitPosterior>
				where TEmitMeanDist : IDistribution<TEmitMean>, CanGetMean<TEmitMean> //, CanGetVariance<TEmitPrec>
				where TEmitPrecDist : IDistribution<TEmitPrec>, CanGetMean<TEmitPrec> //, CanGetVariance<TEmitPrec>
		{
			// fix random seed
			Rand.Restart(12347);

			// set hyperparameters
			var probInitPriorObs = Dirichlet.Uniform(hiddenStates);
			var cptTransPriorObs = Enumerable.Repeat(Dirichlet.Uniform(hiddenStates), hiddenStates).ToArray();
			var emitMeanPriorObs = Enumerable.Repeat(emitPriorMeanFunc(), hiddenStates).ToArray();
			var emitPrecPriorObs = Enumerable.Repeat(emitPriorPrecFunc == null ? default(TEmitPrecDist) : emitPriorPrecFunc(), hiddenStates).ToArray();

			// sample model parameters
			double[][] trans;
			TEmitMean[] emitMeans;
			TEmitPrec[] emitPrecs;
			double[] init;

			SampleModelParameters<TEmitMean, TEmitPrec>(hiddenStates, probInitPriorObs, cptTransPriorObs, 
				emitMeanPriorObs.Cast<Sampleable<TEmitMean>>().ToArray(), 
				emitPriorPrecFunc == null ? null : emitPrecPriorObs.Cast<Sampleable<TEmitPrec>>().ToArray(), 
				out trans, out emitMeans, out emitPrecs, out init);

			// print parameters
			var modelForPrinting = new TModel();
			modelForPrinting.CreateModel(chainLength, hiddenStates, emissionDimension, emitPriorPrecFunc != null, showFactorGraph);
			modelForPrinting.SetParameters(init, trans, emitMeans, emitPrecs);
			Console.WriteLine("parameters:");
			modelForPrinting.PrintParameters();
			Console.WriteLine();

			// create distributions for sampling
			var initDist = new Discrete(init);
			var transDist = new Discrete[hiddenStates];
			for (int i = 0; i < hiddenStates; i++)
			{
				transDist[i] = new Discrete(trans[i]);
			}

			var emitDist = new TEmitDist[hiddenStates];
			for (int i = 0; i < hiddenStates; i++)
			{
				emitDist[i] = emitPriorFunc(emitMeans[i], emitPriorPrecFunc == null ? default(TEmitPrec) : emitPrecs[i]);
			}

			// calculate order of actual states
			var actualStateOrder = ArgSort<TEmitDist, TEmitPosterior>(emitDist);
			Console.WriteLine("actualStateOrder");
			Console.WriteLine(string.Join(",", actualStateOrder));
			Console.WriteLine();

			// sample state and emission data
			TEmit[] emissions;
			int[] actualStates;
			SampleData(chainLength, initDist, transDist, emitDist.Cast<Sampleable<TEmit>>().ToArray(), out emissions, out actualStates);

			Console.WriteLine("sample data:");
			Console.WriteLine(string.Join(",", actualStates));
			Console.WriteLine();
			
			// infer model parameters, states and model evidence given priors and emission data
			var model = new TModel();
			model.CreateModel(chainLength, hiddenStates, emissionDimension, emitPriorPrecFunc != null, showFactorGraph);
			model.SetPriors(probInitPriorObs, cptTransPriorObs, emitMeanPriorObs, emitPrecPriorObs);
			model.ObserveData(emissions);
			model.InitialiseStatesRandomly();
			model.InferPosteriors();
			Console.WriteLine("model likelihood: " + model.modelEvidencePosterior);
			var mapStatesDistr = model.statesPosterior;
			var mapStates = mapStatesDistr.Select(s => s.GetMode()).ToArray();
			Console.WriteLine();

			// print maximum a priori states
			Console.WriteLine("statesMAP");
			Console.WriteLine(string.Join(",", mapStates));
			Console.WriteLine();

			// print posterior distributions
			Console.WriteLine("posteriors");
			model.PrintPosteriors();
			Console.WriteLine();

			// calculate order of MAP states
			int[] mapStateOrder = ArgSort<TEmitMeanDist, TEmitMean>(model.emitMeanPosterior);
			Console.WriteLine("mapStateOrder");
			Console.WriteLine(string.Join(",", mapStateOrder));
			Console.WriteLine();

			// accuracy of MAP estimates
			int correctStates = actualStates.Where((t, i) => actualStateOrder[t] == mapStateOrder[mapStates[i]]).Count();
			Console.WriteLine("correctStates: " + correctStates + " / " + actualStates.Length);
			Console.WriteLine();

			Console.WriteLine("------------------\n");

		}

		/// <summary>
        /// Tests the binary hidden markov model.
        /// </summary>
        public static void TestBinaryHiddenMarkovModel()
        {
            // fix random seed
            Rand.Restart(12347);

            // model size
            const int T = 100;
            const int K = 2;

            // set hyperparameters
            var probInitPriorObs = Dirichlet.Uniform(K);
            var cptTransPriorObs = Enumerable.Repeat(Dirichlet.Uniform(K), K).ToArray();
            var emitPriorObs = Enumerable.Repeat(new Beta(1, 1), K).ToArray();

            // sample model parameters
            var init = probInitPriorObs.Sample().ToArray();
            var trans = new double[K][];
            for (int i = 0; i < K; i++)
            {
                trans[i] = cptTransPriorObs[i].Sample().ToArray();
            }

            var emit = new double[K];
            for (int i = 0; i < K; i++)
            {
                emit[i] = emitPriorObs[i].Sample();
            }

            // print parameters
            var modelForPrinting = new BinaryHMM();
			modelForPrinting.CreateModel(T, K, 1, false);
            modelForPrinting.SetParameters(init, trans, emit, null);
            Console.WriteLine("parameters:");
            modelForPrinting.PrintParameters();
            Console.WriteLine();

            // create distributions for sampling
            var initDist = new Discrete(init);
            var transDist = new Discrete[K];
            for (int i = 0; i < K; i++)
            {
                transDist[i] = new Discrete(trans[i]);
            }
            
            var emitDist = new Bernoulli[K];
            for (int i = 0; i < K; i++)
            {
                emitDist[i] = new Bernoulli(emit[i]);
            }

            // calculate order of actual states
			var actualStateOrder = emitDist.Select((ia, i) => new { ia, i }).OrderBy(pair => pair.ia.GetMean()).Select(pair => pair.i).ToArray();
            Console.WriteLine("actualStateOrder");
            Console.WriteLine(string.Join(",", actualStateOrder));
            Console.WriteLine();

            // sample state and emission data
            var actualStates = new int[T];
            var emissions = new bool[T];
            actualStates[0] = initDist.Sample();
            emissions[0] = emitDist[actualStates[0]].Sample();
            for (int i = 1; i < T; i++)
            {
                actualStates[i] = transDist[actualStates[i - 1]].Sample();
                emissions[i] = emitDist[actualStates[i]].Sample();
            }

            Console.WriteLine("sample data:");
            Console.WriteLine(string.Join(",", actualStates));
            Console.WriteLine();

            // infer model parameters, states and model evidence given priors and emission data
            var model = new BinaryHMM();
			model.CreateModel(T, K, 1, false);
            model.SetPriors(probInitPriorObs, cptTransPriorObs, emitPriorObs, null);
            model.ObserveData(emissions);
            model.InitialiseStatesRandomly();
            model.InferPosteriors();
            Console.WriteLine("model likelihood: " + model.modelEvidencePosterior);
            var mapStatesDistr = model.statesPosterior;
            var mapStates = mapStatesDistr.Select(s => s.GetMode()).ToArray();
            Console.WriteLine();

            // print maximum a priori states
            Console.WriteLine("statesMAP");
            Console.WriteLine(string.Join(",", mapStates));
            Console.WriteLine();

            // print posterior distributions
            Console.WriteLine("posteriors");
            model.PrintPosteriors();
            Console.WriteLine();

            // calculate order of MAP states
			var mapStateOrder = ArgSort<Beta, double>(model.emitMeanPosterior);
            Console.WriteLine(string.Join(",", mapStateOrder));
            Console.WriteLine();

            // accuracy of MAP estimates
            int correctStates = actualStates.Where((t, i) => actualStateOrder[t] == mapStateOrder[mapStates[i]]).Count();
            Console.WriteLine("correctStates: " + correctStates + " / " + actualStates.Length);
            Console.WriteLine();

            Console.WriteLine("------------------\n");
        }

        /// <summary>
        /// Tests the discrete hidden markov model.
        /// </summary>
        public static void TestDiscreteHiddenMarkovModel()
        {
            // fix random seed
            Rand.Restart(12347);

            // model size
            const int T = 100;
            const int K = 2;
            const int E = 5;

            // set hyperparameters
            var probInitPriorObs = Dirichlet.Uniform(K);
            var cptTransPriorObs = Enumerable.Repeat(Dirichlet.Uniform(K), K).ToArray();
            var emitPriorObs = Enumerable.Repeat(Dirichlet.Uniform(E), K).ToArray();

            // sample model parameters
            var init = probInitPriorObs.Sample().ToArray();
            var trans = new double[K][];
            for (int i = 0; i < K; i++)
            {
                trans[i] = cptTransPriorObs[i].Sample().ToArray();
            }

            var emit = new Vector[K];
            for (int i = 0; i < K; i++)
            {
                emit[i] = emitPriorObs[i].Sample();
            }

            // print parameters
            var modelForPrinting = new DiscreteHMM();
			modelForPrinting.CreateModel(T, K, E, false);
			modelForPrinting.SetParameters(init, trans, emit, null);
            Console.WriteLine("parameters:");
            modelForPrinting.PrintParameters();
            Console.WriteLine();

            // create distributions for sampling
            var initDist = new Discrete(init);
            var transDist = new Discrete[K];
            for (int i = 0; i < K; i++)
            {
                transDist[i] = new Discrete(trans[i]);
            }

            var emitDist = new Discrete[K];
            for (int i = 0; i < K; i++)
            {
                emitDist[i] = new Discrete(emit[i]);
            }

            // calculate order of actual states
			// var actualStateOrder = ArgSort<Discrete, double>(emitDist);
			var actualStateOrder = emitDist.Select((ia, i) => new { ia, i }).OrderBy(pair => pair.ia.GetMean()).Select(pair => pair.i).ToArray();
            Console.WriteLine("actualStateOrder");
            Console.WriteLine(string.Join(",", actualStateOrder));
            Console.WriteLine();

            // sample state and emission data
            var actualStates = new int[T];
            var emissions = new int[T];
            actualStates[0] = initDist.Sample();
            emissions[0] = emitDist[actualStates[0]].Sample();
            for (int i = 1; i < T; i++)
            {
                actualStates[i] = transDist[actualStates[i - 1]].Sample();
                emissions[i] = emitDist[actualStates[i]].Sample();
            }

            Console.WriteLine("sample data:");
            Console.WriteLine(string.Join(",", actualStates));
            Console.WriteLine();

            // infer model parameters, states and model evidence given priors and emission data
			var model = new DiscreteHMM();
			model.CreateModel(T, K, E, false);
            model.SetPriors(probInitPriorObs, cptTransPriorObs, emitPriorObs, null);
            model.ObserveData(emissions);
            model.InitialiseStatesRandomly();
            model.InferPosteriors();
            Console.WriteLine("model likelihood: " + model.modelEvidencePosterior);
            var mapStatesDistr = model.statesPosterior;
            var mapStates = mapStatesDistr.Select(s => s.GetMode()).ToArray();
            Console.WriteLine();

            // print maximum a priori states
            Console.WriteLine("statesMAP");
            Console.WriteLine(string.Join(",", mapStates));
            Console.WriteLine();

            // print posterior distributions
            Console.WriteLine("posteriors");
            model.PrintPosteriors();
            Console.WriteLine();
            
            // calculate order of MAP states
			var mapStateOrder = ArgSort<Dirichlet, Vector>(model.emitMeanPosterior);
			// var mapStateOrder = model.emitPosterior.Select((ia, i) => new { ia, i }).OrderBy(pair => pair.ia.GetMean()).Select(pair => pair.i).ToArray();
			Console.WriteLine("mapStateOrder");
            Console.WriteLine(string.Join(",", mapStateOrder));
            Console.WriteLine();

            // accuracy of MAP estimates
            int correctStates = actualStates.Where((t, i) => actualStateOrder[t] == mapStateOrder[mapStates[i]]).Count();
            Console.WriteLine("correctStates: " + correctStates + " / " + actualStates.Length);
            Console.WriteLine();
            
            Console.WriteLine("------------------\n");
        }


		/// <summary>
		/// Arguments the sort.
		/// </summary>
		/// <returns>The sort.</returns>
		/// <param name="emitDist">Emit dist.</param>
		private static int[] ArgSort<TEmitDist, TEmit>(TEmitDist[] emitDist)
		where TEmitDist : CanGetMean<TEmit>
		{
			int[] actualStateOrder;
			if (emitDist[0].GetMean() is IComparable<TEmit>)
			{
				actualStateOrder = emitDist.Select((ia, i) => new {
					ia,
					i
				}).OrderBy(pair => pair.ia.GetMean()).Select(pair => pair.i).ToArray();
			}
			else
				if (emitDist[0].GetMean() is Vector)
				{
					// Must be a vector type
					actualStateOrder = emitDist.Select((ia, i) => new {
						ia,
						i
					}).OrderBy(pair => (pair.ia.GetMean() as Vector).Average()).Select(pair => pair.i).ToArray();
				}
				else
				{
					throw new NotSupportedException();
				}
			return actualStateOrder;
		}

		/// <summary>
		/// Samples the model parameters.
		/// </summary>
		/// <param name="K">K.</param>
		/// <param name="probInitPriorObs">Prob init prior obs.</param>
		/// <param name="cptTransPriorObs">Cpt trans prior obs.</param>
		/// <param name="emitMeanPriorObs">Emit mean prior obs.</param>
		/// <param name="emitPrecPriorObs">Emit prec prior obs.</param>
		/// <param name="trans">Trans.</param>
		/// <param name="emitMeans">Emit means.</param>
		/// <param name="emitPrecs">Emit precs.</param>
		/// <param name="init">Init.</param>
		/// <typeparam name="TEmitMean">The 1st type parameter.</typeparam>
		/// <typeparam name="TEmitPrec">The 2nd type parameter.</typeparam>
		private static void SampleModelParameters<TEmitMean, TEmitPrec>(int K, Sampleable<Vector> probInitPriorObs, IList<Dirichlet> cptTransPriorObs, IList<Sampleable<TEmitMean>> emitMeanPriorObs, IList<Sampleable<TEmitPrec>> emitPrecPriorObs, 
			out double[][] trans, out TEmitMean[] emitMeans, out TEmitPrec[] emitPrecs, out double[] init)
		{
			init = probInitPriorObs.Sample().ToArray();
			trans = new double[K][];
			for (int i = 0; i < K; i++)
			{
				trans[i] = cptTransPriorObs[i].Sample().ToArray();
			}

			emitMeans = new TEmitMean[K];
			for (int i = 0; i < K; i++)
			{
				emitMeans[i] = emitMeanPriorObs[i].Sample();
			}

			if (emitPrecPriorObs == null)
			{
				emitPrecs = default(TEmitPrec[]);
				return;
			}

			emitPrecs = new TEmitPrec[K];
			for (int i = 0; i < K; i++)
			{
				emitPrecs[i] = emitPrecPriorObs[i].Sample();
			}
		}

		/// <summary>
		/// Samples the data.
		/// </summary>
		/// <param name="T">T.</param>
		/// <param name="initDist">Init dist.</param>
		/// <param name="transDist">Trans dist.</param>
		/// <param name="emitDist">Emit dist.</param>
		/// <param name="emissions">Emissions.</param>
		/// <param name="actualStates">Actual states.</param>
		/// <typeparam name="TEmit">The 1st type parameter.</typeparam>
		private static void SampleData<TEmit>(int T, Sampleable<int> initDist, IList<Sampleable<int>> transDist, IList<Sampleable<TEmit>> emitDist, out TEmit[] emissions, out int[] actualStates)
		{
			actualStates = new int[T];
			emissions = new TEmit[T];
			actualStates[0] = initDist.Sample();
			emissions[0] = emitDist[actualStates[0]].Sample();
			for (int i = 1; i < T; i++)
			{
				actualStates[i] = transDist[actualStates[i - 1]].Sample();
				emissions[i] = emitDist[actualStates[i]].Sample();
			}
		}
    }
}