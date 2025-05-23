import { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';

interface PredResp {
  Year: number[];
  y_hist: (number | null)[];
  y_pred: (number | null)[];
}

interface FutureResp {
  Year: number[];
  y_pred: number[];
}

interface Props {
  model: string;
  yTitle: string;
  future?: FutureResp | null;
}

export default function ForecastChart({ model, yTitle, future }: Props) {
  const [pred, setPred] = useState<PredResp | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/preds/${model}`)
      .then(r => r.json())
      .then(d => {
        if (d.status === 'error') {
          setError(d.detail);
          setPred(null);
        } else {
          setError(null);
          setPred(d as PredResp);
        }
      })
      .catch(err => {
        setError(err.message);
        setPred(null);
      });
  }, [model]);

  if (error) return <p style={{ color: 'red' }}>Ошибка: {error}</p>;
  if (!pred) return null;

  const data = [
    {
      x: pred.Year,
      y: pred.y_hist,
      mode: 'lines',
      name: 'Исторические',
      line: { color: 'black' },
    },
    {
      x: pred.Year,
      y: pred.y_pred,
      mode: 'lines',
      name: `${model.toUpperCase()} (test)`,
      line: { dash: 'dash' },
    },
  ];

  if (future) {
    data.push({
      x: future.Year,
      y: future.y_pred,
      mode: 'lines',
      name: `${model.toUpperCase()} (forecast)`,
      line: { dash: 'dot'},
    });
  }

  return (
    <Plot
      data={data}
      layout={{
        title: yTitle,
        xaxis: { title: 'Year' },
        yaxis: { title: yTitle },
        legend: { orientation: 'h', x: 0.25, y: 1.15 },
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
        margin: { t: 60, r: 40, l: 60, b: 50 },
      }}
      style={{ width: '100%', height: 420 }}
      config={{ responsive: true }}
    />
  );
}
