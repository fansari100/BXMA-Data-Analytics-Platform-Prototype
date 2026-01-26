"use client";

import "katex/dist/katex.min.css";
import { InlineMath, BlockMath } from "react-katex";

interface MathProps {
  children: string;
  block?: boolean;
  className?: string;
}

/**
 * Math component for rendering LaTeX formulas using KaTeX.
 * 
 * @param children - LaTeX string to render
 * @param block - If true, renders as block math (centered, larger)
 * @param className - Additional CSS classes
 * 
 * @example
 * // Inline math
 * <Math>r_p = \sum_{i=1}^{n} w_i r_i</Math>
 * 
 * // Block math (centered)
 * <Math block>VaR_\alpha = -\inf\{l : P(L > l) \leq 1-\alpha\}</Math>
 */
export function Math({ children, block = false, className = "" }: MathProps) {
  // Sanitize the LaTeX string - remove extra escaping that might come from JSON
  const sanitizedLatex = children
    .replace(/\\\\([a-zA-Z])/g, "\\$1") // Fix double backslashes before commands
    .replace(/\\\\/g, "\\"); // Fix remaining double backslashes

  try {
    if (block) {
      return (
        <div className={`my-4 ${className}`}>
          <BlockMath math={sanitizedLatex} />
        </div>
      );
    }
    return (
      <span className={className}>
        <InlineMath math={sanitizedLatex} />
      </span>
    );
  } catch (error) {
    // Fallback to code display if KaTeX fails to parse
    console.error("KaTeX rendering error:", error);
    return (
      <code className={`font-mono text-accent-cyan ${className}`}>
        {children}
      </code>
    );
  }
}

/**
 * MathBlock is an alias for Math with block=true for convenience
 */
export function MathBlock({ children, className = "" }: Omit<MathProps, "block">) {
  return <Math block className={className}>{children}</Math>;
}

/**
 * Common financial formulas as constants for reuse
 */
export const FORMULAS = {
  // VaR formulas
  VAR_DEFINITION: "VaR_\\alpha = -\\inf\\{l : P(L > l) \\leq 1-\\alpha\\}",
  VAR_PARAMETRIC: "VaR = \\mu - z_\\alpha \\cdot \\sigma",
  VAR_SCALED: "VaR_T = VaR_1 \\cdot \\sqrt{T}",
  
  // Portfolio return
  PORTFOLIO_RETURN: "r_p = \\sum_{i=1}^{n} w_i r_i = \\mathbf{w}^\\prime \\mathbf{r}",
  
  // Sharpe Ratio
  SHARPE_RATIO: "SR = \\frac{E[R_p] - R_f}{\\sigma_p} = \\frac{\\bar{r}_p - r_f}{\\sigma_p}",
  
  // CVaR / Expected Shortfall
  CVAR: "CVaR_\\alpha = E[L | L > VaR_\\alpha] = \\frac{1}{1-\\alpha} \\int_\\alpha^1 VaR_u \\, du",
  
  // Covariance
  COVARIANCE: "\\Sigma_{ij} = Cov(r_i, r_j) = E[(r_i - \\mu_i)(r_j - \\mu_j)]",
  EWMA_COVARIANCE: "\\Sigma_t = \\lambda \\Sigma_{t-1} + (1-\\lambda) r_{t-1} r_{t-1}^\\prime",
  
  // Portfolio Variance
  PORTFOLIO_VARIANCE: "\\sigma_p^2 = \\mathbf{w}^\\prime \\Sigma \\mathbf{w}",
  
  // Z-score
  Z_SCORE: "z_\\alpha = \\Phi^{-1}(\\alpha)",
  
  // Mean
  SAMPLE_MEAN: "\\bar{r} = \\frac{1}{T} \\sum_{t=1}^{T} r_t",
  
  // Volatility
  SAMPLE_STD: "\\sigma = \\sqrt{\\frac{1}{T-1} \\sum_{t=1}^{T} (r_t - \\bar{r})^2}",
  
  // Annualization
  ANNUALIZED_RETURN: "r_{ann} = \\bar{r}_{daily} \\times 252",
  ANNUALIZED_VOL: "\\sigma_{ann} = \\sigma_{daily} \\times \\sqrt{252}",
};

export default Math;
