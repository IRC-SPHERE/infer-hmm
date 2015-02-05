﻿//
// BinaryHMM.cs
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

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Models;

	/// <summary>
	/// Binary HMM.
	/// </summary>
	public class BinaryHMM : HMM<bool, Beta, double, Beta, double>
	{
		/// <summary>
		/// The Emission function.
		/// </summary>
		/// <returns>The function.</returns>
		/// <param name="means">Means.</param>
		/// <param name="precs">Precs.</param>
		/// <param name="state">State.</param>
		public override Variable<bool> EmissionFunction(VariableArray<double> means, VariableArray<double> precs, Variable<int> state)
		{
			return Variable.Bernoulli(means[state]);
		}
	}
}
