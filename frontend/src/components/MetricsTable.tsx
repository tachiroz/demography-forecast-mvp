import { useEffect, useState } from 'react';

interface Metrics {
  MAE:  number;
  MSE:  number;
  MAPE: number;
  R2:   number;
  // если бекенд пришлёт status/error — эти поля не будут присутствовать
}

export default function MetricsTable({ model }: { model: string }) {
  const [data, setData] = useState<Metrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  // ── загружаем метрики при смене model ───────────────────────
  useEffect(() => {
    fetch(`/metrics/${model}`)
      .then(r => r.json())
      .then(resp => {
        if (resp.status === 'error') {
          setError(resp.detail || 'Backend error');
          setData(null);
        } else {
          setError(null);
          setData(resp as Metrics);
        }
      })
      .catch(err => {
        setError(err.message);
        setData(null);
      });
  }, [model]);

  // ── вывод ───────────────────────────────────────────────────
  if (error) return <p style={{ color: 'red' }}>Ошибка: {error}</p>;
  if (!data)  return null; // ничего, пока метрик нет

  return (
    <table style={{ borderCollapse: 'collapse', marginTop: 16 }}>
      <tbody>
        {Object.entries(data).map(([key, val]) => (
          <tr key={key}>
            <td style={{ border: '1px solid #aaa', padding: 4, fontWeight: 600 }}>
              {key}
            </td>
            <td style={{ border: '1px solid #aaa', padding: 4 }}>
              {typeof val === 'number' ? val.toFixed(2) : String(val)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}