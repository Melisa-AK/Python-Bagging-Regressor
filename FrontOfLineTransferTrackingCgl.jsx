import { translate } from "shared/base/translation";
import Translation from "translation.js";

const t = translate(Translation);

const FrontOfLineTransferTrackingCgl = () => {
  // Grid data - görseldeki gibi farklı bölümler
  const mainGridData = Array.from({ length: 8 }, (_, row) =>
    Array.from({ length: 30 }, (_, col) => ({
      id: `${row}-${col}`,
      number: Math.floor(Math.random() * 999).toString().padStart(3, '0'),
      active: Math.random() > 0.5
    }))
  );

  // Alt bölüm için daha küçük grid
  const bottomGridData = Array.from({ length: 4 }, (_, row) =>
    Array.from({ length: 15 }, (_, col) => ({
      id: `bottom-${row}-${col}`,
      number: Math.floor(Math.random() * 999).toString().padStart(3, '0'),
      active: Math.random() > 0.6
    }))
  );

  return (
    <div className="w-full h-screen bg-gray-300">
      {/* Header */}
      <div className="bg-blue-600 text-white p-3 text-center">
        <h1 className="text-xl font-bold">IA1 THE ENTRY AREA LOGIST</h1>
        <div className="text-sm mt-1">2025/1/18 11:22:37</div>
      </div>

      <div className="p-6">
        {/* Ana Grid Bölümü */}
        <div className="bg-white border-2 border-gray-600 p-4 mb-6">
          <div className="grid gap-1 mb-4" style={{ gridTemplateColumns: 'repeat(30, 1fr)' }}>
            {mainGridData.map((row, rowIndex) =>
              row.map((item, colIndex) => (
                <div
                  key={`${rowIndex}-${colIndex}`}
                  className={`
                    border h-10 flex items-center justify-center text-xs font-mono
                    ${item.active ? 'bg-blue-400 border-blue-700 text-white font-bold' : 'bg-gray-200 border-gray-500'}
                  `}
                >
                  {item.active && item.number}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Yol Ayrımı - Ortada boşluk */}
        <div className="flex gap-8 mb-6">
          {/* Sol Alt Grid */}
          <div className="flex-1 bg-white border-2 border-gray-600 p-4">
            <div className="grid gap-1" style={{ gridTemplateColumns: 'repeat(15, 1fr)' }}>
              {bottomGridData.map((row, rowIndex) =>
                row.map((item, colIndex) => (
                  <div
                    key={`left-${rowIndex}-${colIndex}`}
                    className={`
                      border h-10 flex items-center justify-center text-xs font-mono
                      ${item.active ? 'bg-green-400 border-green-700 text-white font-bold' : 'bg-gray-200 border-gray-500'}
                    `}
                  >
                    {item.active && item.number}
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Ortada Boşluk / Yol Ayrımı */}
          <div className="w-32 flex flex-col items-center justify-center">
            <div className="text-center">
              <div className="text-lg font-bold mb-2">↓</div>
              <div className="text-sm bg-yellow-400 px-2 py-1 rounded">AYRIM</div>
              <div className="text-lg font-bold mt-2">↓</div>
            </div>
          </div>

          {/* Sağ Alt Grid */}
          <div className="flex-1 bg-white border-2 border-gray-600 p-4">
            <div className="grid gap-1" style={{ gridTemplateColumns: 'repeat(15, 1fr)' }}>
              {bottomGridData.map((row, rowIndex) =>
                row.map((item, colIndex) => (
                  <div
                    key={`right-${rowIndex}-${colIndex}`}
                    className={`
                      border h-10 flex items-center justify-center text-xs font-mono
                      ${item.active ? 'bg-cyan-400 border-cyan-700 text-white font-bold' : 'bg-gray-200 border-gray-500'}
                    `}
                  >
                    {item.active && item.number}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Konveyör Yön Ok İşaretleri */}
        <div className="flex justify-center mb-4">
          <svg width="800" height="60" className="border bg-white">
            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#1e40af" />
              </marker>
            </defs>
            
            {/* Ana konveyör hatları */}
            <line x1="50" y1="20" x2="750" y2="20" stroke="#1e40af" strokeWidth="4" markerEnd="url(#arrowhead)" />
            <line x1="50" y1="40" x2="400" y2="40" stroke="#16a34a" strokeWidth="4" markerEnd="url(#arrowhead)" />
            <line x1="450" y1="40" x2="750" y2="40" stroke="#0891b2" strokeWidth="4" markerEnd="url(#arrowhead)" />
            
            <text x="400" y="15" textAnchor="middle" className="text-sm font-bold">ANA HAT</text>
            <text x="225" y="55" textAnchor="middle" className="text-xs font-bold">SOL HAT</text>
            <text x="600" y="55" textAnchor="middle" className="text-xs font-bold">SAĞ HAT</text>
          </svg>
        </div>

        {/* Alt Kontrol Butonları */}
        <div className="flex justify-center gap-4">
          <button className="px-6 py-3 bg-gray-500 text-white rounded-lg font-semibold">RETURN</button>
          <button className="px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold">CGL1</button>
          <button className="px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold">CGL2</button>
          <button className="px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold">CGL3</button>
          <button className="px-6 py-3 bg-yellow-500 text-white rounded-lg font-semibold">EXIT</button>
        </div>
      </div>
    </div>
  );
};

export default FrontOfLineTransferTrackingCgl;