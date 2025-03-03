import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Brain } from 'lucide-react';
import AttentionVisualization from './components/AttentionVisualization';
import TransformerDiagram from './components/TransformerDiagram';

function App() {
  const [input, setInput] = useState('The cat sat on the mat');
  const [tokens, setTokens] = useState([]);
  const [attentionWeights, setAttentionWeights] = useState([]);
  const [selectedHead, setSelectedHead] = useState(0);
  const [modelDimensions, setModelDimensions] = useState({
    d_model: 64,
    nhead: 4,
    head_dim: 16,
    dim_feedforward: 128
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  // Check if the API is running
  useEffect(() => {
    const checkApiStatus = async () => {
      try {
        await axios.get('http://localhost:5000/api/health');
        setApiStatus('running');
      } catch (error) {
        setApiStatus('not-running');
      }
    };

    checkApiStatus();
    const interval = setInterval(checkApiStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const processInput = async () => {
    if (!input.trim()) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('http://localhost:5000/api/process', {
        input: input
      });
      
      setTokens(response.data.input_tokens);
      setAttentionWeights(response.data.attention_weights);
      setModelDimensions(response.data.model_dimensions);
    } catch (err) {
      setError('Failed to process input. Make sure the API server is running.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleHeadChange = (e) => {
    setSelectedHead(parseInt(e.target.value));
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-indigo-600 text-white p-4 shadow-md">
        <div className="container mx-auto flex items-center">
          <Brain className="mr-2" size={24} />
          <h1 className="text-xl font-bold">Transformer Visualization</h1>
        </div>
      </header>

      <main className="container mx-auto py-8 px-4">
        {apiStatus === 'not-running' && (
          <div className="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 mb-6 rounded">
            <p className="font-bold">API Not Running</p>
            <p>Please start the Python API server by running: <code className="bg-gray-200 px-2 py-1 rounded">npm run start-api</code></p>
          </div>
        )}

        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-lg font-semibold mb-4">Input Text</h2>
          <div className="flex flex-col md:flex-row gap-4">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className="flex-grow p-2 border rounded focus:outline-none focus:ring-2 focus:ring-indigo-500"
              placeholder="Enter text to process..."
            />
            <button
              onClick={processInput}
              disabled={isLoading || apiStatus !== 'running'}
              className={`px-4 py-2 rounded font-medium ${
                isLoading || apiStatus !== 'running'
                  ? 'bg-gray-300 cursor-not-allowed'
                  : 'bg-indigo-600 text-white hover:bg-indigo-700'
              }`}
            >
              {isLoading ? 'Processing...' : 'Process'}
            </button>
          </div>
          {error && <p className="text-red-500 mt-2">{error}</p>}
        </div>

        {tokens.length > 0 && attentionWeights.length > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div>
              <div className="bg-white rounded-lg shadow-md p-6 mb-6">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-lg font-semibold">Attention Visualization</h2>
                  <div className="flex items-center">
                    <label htmlFor="head-select" className="mr-2 text-sm">Head:</label>
                    <select
                      id="head-select"
                      value={selectedHead}
                      onChange={handleHeadChange}
                      className="p-1 border rounded"
                    >
                      {Array.from({ length: modelDimensions.nhead }, (_, i) => (
                        <option key={i} value={i}>
                          {i + 1}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <AttentionVisualization
                  attentionWeights={attentionWeights}
                  tokens={tokens}
                  selectedHead={selectedHead}
                />
              </div>

              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-lg font-semibold mb-4">Tokens</h2>
                <div className="grid grid-cols-2 gap-2">
                  {tokens.map((token, index) => (
                    <div key={index} className="bg-gray-100 p-2 rounded flex justify-between">
                      <span className="font-medium">{index}:</span>
                      <span className="text-indigo-600">{token}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div>
              <TransformerDiagram modelDimensions={modelDimensions} />
              
              <div className="bg-white rounded-lg shadow-md p-6 mt-6">
                <h2 className="text-lg font-semibold mb-4">How It Works</h2>
                <div className="space-y-3 text-sm">
                  <p>
                    <span className="font-medium">Self-Attention:</span> Each token attends to all other tokens in the sequence, 
                    allowing the model to capture relationships regardless of distance.
                  </p>
                  <p>
                    <span className="font-medium">Multi-Head Attention:</span> The model uses {modelDimensions.nhead} parallel attention 
                    heads, each focusing on different aspects of the relationships between tokens.
                  </p>
                  <p>
                    <span className="font-medium">Attention Weights:</span> The heatmap shows how much each token (y-axis) 
                    attends to other tokens (x-axis). Darker colors indicate stronger attention.
                  </p>
                  <p>
                    <span className="font-medium">Feed-Forward Network:</span> After attention, each token's representation 
                    is processed independently through a feed-forward neural network.
                  </p>
                  <p>
                    <span className="font-medium">Residual Connections:</span> The model uses residual connections (+ arrows) 
                    to help with gradient flow during training.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="bg-gray-100 p-4 border-t">
        <div className="container mx-auto text-center text-gray-600 text-sm">
          Transformer Visualization - Based on "Attention Is All You Need" paper
        </div>
      </footer>
    </div>
  );
}

export default App;