from pathlib import Path
import shutil

paper = Path(__file__).resolve().parents[2]
record = Path(__file__).resolve().parent
old = 'figures/generated_current_20261102/'
new = 'figures/journal_style_20261102/'
for filename in ('three_frame_estimation.png', 'forecast_application.png'):
    assert (paper / new / filename).is_file(), filename
for name in ('main.tex', 'main_zh.md'):
    p = paper / name
    text = p.read_text(encoding='utf-8')
    assert text.count(old) == 2, (name, text.count(old))
    backup = record / (name.replace('.', '_') + '_before.txt')
    assert not backup.exists(), 'Integration already attempted; do not overwrite backups'
    shutil.copyfile(p, backup)
    text = text.replace(old, new)
    if name == 'main.tex':
        text = text.replace('Three-frame COT-assisted GHI estimation. Observed AGRI', 'Three-frame COT-assisted GHI estimation. (a) Framewise COT retrieval from observed imagery. (b) Irradiance estimation from the aligned local sequence. Observed AGRI', 1)
        text = text.replace('AI-generated visual prototype; cloud textures, maps and station icons are illustrative, not observational or forecast results.', 'AI-generated schematic; cloud textures and field tiles are illustrative, not observational or forecast results.', 1)
    else:
        text = text.replace('图2：真实三帧COT辅助GHI估计。冻结反演器', '图2：真实三帧COT辅助GHI估计。(a)逐帧反演观测图像的COT；(b)利用对齐的局地序列估计辐照度。冻结反演器', 1)
        text = text.replace('本图为AI生成的视觉效果稿，云纹理、地图与站点图标均为示意，不是观测或预报结果。', '本图为AI生成的示意图，云纹理与场图块不是观测或预报结果。', 1)
    p.write_text(text, encoding='utf-8')
print('Updated exactly two figure references per language and synchronized panel captions.')
