import { useState } from 'react';

export default function ModelForm({ onTrained }: { onTrained: () => void }) {
  const [model, setModel] = useState('sarimax');
  const [loading, setLoading] = useState(false);

  const train = async () => {
    setLoading(true);
    await fetch('/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, params: {} }),
    });
    setLoading(false);
    onTrained();          // сообщаем родителю: можно тянуть метрики и график
  };

  return (
    <div style={{ marginBottom: 12 }}>
      <label style={{ marginRight: 8 }}>Model:</label>
      <select value={model} onChange={e => setModel(e.target.value)}>
        <option value="sarimax">SARIMAX</option>
        {/* позже добавим prophet, xgb, cat */}
      </select>
      <button style={{ marginLeft: 12 }} onClick={train} disabled={loading}>
        {loading ? 'Training…' : 'Train'}
      </button>
    </div>
  );
}
