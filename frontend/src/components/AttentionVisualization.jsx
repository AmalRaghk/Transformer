import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

const AttentionVisualization = ({ attentionWeights, tokens, selectedHead }) => {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!attentionWeights || attentionWeights.length === 0 || !tokens.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Extract attention weights for the selected head (first batch)
    const weights = attentionWeights[0][selectedHead];
    
    const margin = { top: 20, right: 20, bottom: 90, left: 90 };
    const width = 500 - margin.left - margin.right;
    const height = 500 - margin.top - margin.bottom;

    const g = svg
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Create color scale
    const colorScale = d3.scaleSequential(d3.interpolateBlues)
      .domain([0, d3.max(weights.flat()) || 1]);

    // Create scales
    const xScale = d3.scaleBand()
      .domain(tokens.map((_, i) => i.toString()))
      .range([0, width])
      .padding(0.05);

    const yScale = d3.scaleBand()
      .domain(tokens.map((_, i) => i.toString()))
      .range([0, height])
      .padding(0.05);

    // Create the heatmap cells
    for (let i = 0; i < weights.length; i++) {
      for (let j = 0; j < weights[i].length; j++) {
        g.append("rect")
          .attr("x", xScale(j.toString()) || 0)
          .attr("y", yScale(i.toString()) || 0)
          .attr("width", xScale.bandwidth())
          .attr("height", yScale.bandwidth())
          .attr("fill", colorScale(weights[i][j]))
          .attr("stroke", "#fff")
          .attr("stroke-width", 0.5)
          .append("title")
          .text(`From: ${tokens[j]}, To: ${tokens[i]}, Weight: ${weights[i][j].toFixed(4)}`);
      }
    }

    // Add x-axis labels (source tokens)
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .selectAll("text")
      .data(tokens)
      .enter()
      .append("text")
      .attr("x", (_, i) => (xScale(i.toString()) || 0) + xScale.bandwidth() / 2)
      .attr("y", 10)
      .attr("dy", ".35em")
      .attr("transform", (_, i) => {
        const x = (xScale(i.toString()) || 0) + xScale.bandwidth() / 2;
        return `rotate(-45, ${x}, 10)`;
      })
      .style("text-anchor", "end")
      .style("font-size", "12px")
      .text(d => d);

    // Add y-axis labels (target tokens)
    g.append("g")
      .selectAll("text")
      .data(tokens)
      .enter()
      .append("text")
      .attr("x", -10)
      .attr("y", (_, i) => (yScale(i.toString()) || 0) + yScale.bandwidth() / 2)
      .attr("dy", ".35em")
      .style("text-anchor", "end")
      .style("font-size", "12px")
      .text(d => d);

    // Add axis titles
    g.append("text")
      .attr("x", width / 2)
      .attr("y", height + margin.bottom - 10)
      .style("text-anchor", "middle")
      .style("font-size", "14px")
      .style("font-weight", "bold")
      .text("Source Tokens (Keys)");

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -height / 2)
      .attr("y", -margin.left + 20)
      .style("text-anchor", "middle")
      .style("font-size", "14px")
      .style("font-weight", "bold")
      .text("Target Tokens (Queries)");

    // Add title
    g.append("text")
      .attr("x", width / 2)
      .attr("y", -margin.top / 2)
      .style("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text(`Attention Weights - Head ${selectedHead + 1}`);

  }, [attentionWeights, tokens, selectedHead]);

  return (
    <div className="border rounded-lg p-4 bg-white shadow-md">
      <svg ref={svgRef}></svg>
    </div>
  );
};

export default AttentionVisualization;