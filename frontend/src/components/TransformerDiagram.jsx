import React from 'react';
import { ArrowRight, ArrowDown, Plus, Divide } from 'lucide-react';

const TransformerDiagram = ({ modelDimensions }) => {
  const { d_model, nhead, head_dim, dim_feedforward } = modelDimensions;

  return (
    <div className="border rounded-lg p-6 bg-white shadow-md">
      <h3 className="text-lg font-semibold mb-4">Transformer Architecture</h3>
      
      <div className="flex flex-col items-center">
        {/* Input */}
        <div className="w-40 h-10 bg-blue-100 rounded flex items-center justify-center mb-2 border border-blue-300">
          <span className="text-sm">Input Embedding</span>
        </div>
        
        <ArrowDown className="my-1 text-gray-500" size={20} />
        
        {/* Multi-Head Attention */}
        <div className="w-64 bg-yellow-50 rounded-lg p-4 border border-yellow-200 mb-2">
          <div className="text-sm font-medium text-center mb-2">Multi-Head Attention</div>
          
          <div className="flex justify-between mb-2">
            <div className="flex flex-col items-center">
              <div className="w-16 h-8 bg-yellow-100 rounded flex items-center justify-center text-xs border border-yellow-300">
                Q ({d_model})
              </div>
              <ArrowDown className="my-1 text-gray-500" size={16} />
              <div className="w-16 h-8 bg-yellow-200 rounded flex items-center justify-center text-xs border border-yellow-300">
                Split ({nhead})
              </div>
            </div>
            
            <div className="flex flex-col items-center">
              <div className="w-16 h-8 bg-yellow-100 rounded flex items-center justify-center text-xs border border-yellow-300">
                K ({d_model})
              </div>
              <ArrowDown className="my-1 text-gray-500" size={16} />
              <div className="w-16 h-8 bg-yellow-200 rounded flex items-center justify-center text-xs border border-yellow-300">
                Split ({nhead})
              </div>
            </div>
            
            <div className="flex flex-col items-center">
              <div className="w-16 h-8 bg-yellow-100 rounded flex items-center justify-center text-xs border border-yellow-300">
                V ({d_model})
              </div>
              <ArrowDown className="my-1 text-gray-500" size={16} />
              <div className="w-16 h-8 bg-yellow-200 rounded flex items-center justify-center text-xs border border-yellow-300">
                Split ({nhead})
              </div>
            </div>
          </div>
          
          <div className="flex justify-center items-center mb-2">
            <ArrowDown className="text-gray-500" size={16} />
            <div className="mx-2 w-6 text-center">×</div>
            <ArrowDown className="text-gray-500" size={16} />
            <div className="mx-2 w-6 text-center">×</div>
            <ArrowDown className="text-gray-500" size={16} />
          </div>
          
          <div className="w-full h-10 bg-yellow-300 rounded flex items-center justify-center mb-2 text-sm border border-yellow-400">
            Scaled Dot-Product Attention
          </div>
          
          <div className="flex justify-center">
            <ArrowDown className="text-gray-500" size={16} />
          </div>
          
          <div className="w-full h-8 bg-yellow-200 rounded flex items-center justify-center text-xs mt-2 border border-yellow-300">
            Concat Heads & Linear ({d_model})
          </div>
        </div>
        
        <div className="flex items-center">
          <ArrowDown className="mx-2 text-gray-500" size={20} />
          <Plus className="mx-2 text-gray-500" size={20} />
          <ArrowRight className="mx-2 text-gray-500" size={20} />
        </div>
        
        {/* Layer Norm */}
        <div className="w-40 h-10 bg-purple-100 rounded flex items-center justify-center my-2 border border-purple-300">
          <span className="text-sm">Layer Norm</span>
        </div>
        
        <ArrowDown className="my-1 text-gray-500" size={20} />
        
        {/* Feed Forward */}
        <div className="w-64 bg-green-50 rounded-lg p-4 border border-green-200 mb-2">
          <div className="text-sm font-medium text-center mb-2">Feed Forward Network</div>
          
          <div className="w-full h-8 bg-green-100 rounded flex items-center justify-center text-xs mb-2 border border-green-300">
            Linear ({d_model} → {dim_feedforward})
          </div>
          
          <ArrowDown className="mx-auto my-1 text-gray-500" size={16} />
          
          <div className="w-full h-8 bg-green-100 rounded flex items-center justify-center text-xs mb-2 border border-green-300">
            ReLU
          </div>
          
          <ArrowDown className="mx-auto my-1 text-gray-500" size={16} />
          
          <div className="w-full h-8 bg-green-100 rounded flex items-center justify-center text-xs border border-green-300">
            Linear ({dim_feedforward} → {d_model})
          </div>
        </div>
        
        <div className="flex items-center">
          <ArrowDown className="mx-2 text-gray-500" size={20} />
          <Plus className="mx-2 text-gray-500" size={20} />
          <ArrowRight className="mx-2 text-gray-500" size={20} />
        </div>
        
        {/* Layer Norm */}
        <div className="w-40 h-10 bg-purple-100 rounded flex items-center justify-center my-2 border border-purple-300">
          <span className="text-sm">Layer Norm</span>
        </div>
        
        <ArrowDown className="my-1 text-gray-500" size={20} />
        
        {/* Output */}
        <div className="w-40 h-10 bg-blue-100 rounded flex items-center justify-center border border-blue-300">
          <span className="text-sm">Output</span>
        </div>
      </div>
      
      <div className="mt-6 text-sm text-gray-600">
        <div><span className="font-medium">Model Dimension (d_model):</span> {d_model}</div>
        <div><span className="font-medium">Number of Heads (nhead):</span> {nhead}</div>
        <div><span className="font-medium">Head Dimension:</span> {head_dim}</div>
        <div><span className="font-medium">Feed-Forward Dimension:</span> {dim_feedforward}</div>
      </div>
    </div>
  );
};

export default TransformerDiagram;