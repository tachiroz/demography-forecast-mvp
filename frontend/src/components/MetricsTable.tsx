import { useEffect, useState } from 'react';

interface Metrics {
  MAE:  number;
  MSE:  number;
  MAPE: number;
  R2:   number;
}

export default function MetricsTable({ model }: { model: string }) {
  const [data, setData] = useState<Metrics | null>(null);

  useEffect(() => {
    fetch(`/metrics/${model}`)
      .then(r => r.json())
      .then(setData)
      .catch(() => setData(null));
  }, [model]);

  if (!data) return null;

  return (
    <table style={{ borderCollapse: 'collapse', marginTop: 16 }}>
      <tbody>
        {Object.entries(data).map(([k, v]) => (
          <tr key={k}>
            <td style={{ border: '1px solid #aaa', padding: 4, fontWeight: 600 }}>{k}</td>
            <td style={{ border: '1px solid #aaa', padding: 4 }}>{v.toFixed(2)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
