import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

export function Math({ children }: { children: string }) {
  return <InlineMath math={children} />;
}

export function MathBlock({ children }: { children: string }) {
  return (
    <div className="math-block">
      <BlockMath math={children} />
    </div>
  );
}
