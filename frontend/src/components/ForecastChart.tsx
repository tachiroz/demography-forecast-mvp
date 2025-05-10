import { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';

export default function ForecastChart({ model }: { model: string }) {
  const [pred, setPred] = useState<any | null>(null);

  useEffect(() => {
    fetch(`/preds/${model}`)
      .then(r => r.json())
      .then(setPred)
      .catch(() => setPred(null));
  }, [model]);

  if (!pred) return null;

  return (
    <Plot
      data={[
        {
          x: pred.Year,
          y: pred.y_true,
          mode: 'lines',
          name: 'Исторические',
          line: { color: 'black' },
        },
        {
          x: pred.Year,
          y: pred.y_pred,
          mode: 'lines',
          name: model.toUpperCase(),
          line: { dash: 'dash' },
        },
      ]}
      layout={{
        title: `Forecast – ${model}`,
        xaxis: { title: 'Year' },
        yaxis: { title: 'Births' },
        shapes: [
          {
            type: 'line',
            x0: 2021.5,
            x1: 2021.5,
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: { dash: 'dot', color: 'grey' },
          },
        ],
      }}
      style={{ width: '100%', height: 450 }}
    />
  );
}
