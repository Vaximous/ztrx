//! The frame-rate overlay: our own text drawn into the swap chain's back
//! buffer just before it is presented (never into title targets, so saved
//! frames do not have it). It needs no shader: the text is a 5x7 pixel
//! font whose lit cells are rectangles cleared to a colour
//! (`ClearRenderTargetView` with a list of rectangles), on a dark box
//! cleared the same way. `VICTORY_FPS_OVERLAY=0` turns it off.

/// Whether the overlay is drawn (on unless `VICTORY_FPS_OVERLAY=0`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FPS_OVERLAY").is_none_or(|v| v != "0"))
}

/// A rectangle in back-buffer pixels, right and bottom exclusive (as a
/// `D3D12_RECT`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rect {
    pub(crate) left: i32,
    pub(crate) top: i32,
    pub(crate) right: i32,
    pub(crate) bottom: i32,
}

/// Glyph cells across and down.
const WIDTH: i32 = 5;
const HEIGHT: i32 = 7;
/// Cells from one glyph to the next, and from one line to the next.
const ADVANCE: i32 = WIDTH + 1;
const LINE: i32 = HEIGHT + 2;
/// Cells of dark box around the text.
const PAD: i32 = 2;

/// The rows of a glyph, top first, the leftmost cell the highest of five
/// bits; `None` for a character the font lacks (drawn as a space).
fn glyph(c: char) -> Option<[u8; 7]> {
    Some(match c {
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        '.' => [0, 0, 0, 0, 0, 0b01100, 0b01100],
        '-' => [0, 0, 0, 0b11111, 0, 0, 0],
        'c' => [0, 0, 0b01110, 0b10000, 0b10000, 0b10001, 0b01110],
        'f' => [0b00110, 0b01001, 0b01000, 0b11100, 0b01000, 0b01000, 0b01000],
        'g' => [0, 0b01111, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110],
        'm' => [0, 0, 0b11010, 0b10101, 0b10101, 0b10001, 0b10001],
        'p' => [0, 0, 0b11110, 0b10001, 0b11110, 0b10000, 0b10000],
        's' => [0, 0, 0b01110, 0b10000, 0b01110, 0b00001, 0b11110],
        'u' => [0, 0, 0b10001, 0b10001, 0b10001, 0b10011, 0b01101],
        _ => return None,
    })
}

/// The cell size for a back buffer this tall: 4 pixels at 1440 lines,
/// 3 at 1080, never under 2.
pub(crate) fn scale(height: u32) -> i32 {
    (height / 360).max(2) as i32
}

/// The overlay for `lines` with its box's corner at (`x`, `y`) and cells
/// `scale` pixels wide, kept inside a `width` x `height` target: the dark
/// box, then the lit cells (a run of lit cells in a row is one rectangle).
/// `None` if there is no text or nothing of it is inside the target.
pub(crate) fn layout(lines: &[&str], x: i32, y: i32, scale: i32, width: i32, height: i32) -> Option<(Rect, Vec<Rect>)> {
    let columns = lines.iter().map(|l| l.chars().count() as i32).max().unwrap_or(0);
    if columns == 0 || scale <= 0 {
        return None;
    }
    let clip = |r: Rect| {
        let r = Rect { left: r.left.max(0), top: r.top.max(0), right: r.right.min(width), bottom: r.bottom.min(height) };
        (r.left < r.right && r.top < r.bottom).then_some(r)
    };
    let backing = clip(Rect {
        left: x,
        top: y,
        right: x + (2 * PAD + columns * ADVANCE - 1) * scale,
        bottom: y + (2 * PAD + lines.len() as i32 * LINE - (LINE - HEIGHT)) * scale,
    })?;
    let mut lit = Vec::new();
    for (row_of_text, line) in lines.iter().enumerate() {
        let top = y + (PAD + row_of_text as i32 * LINE) * scale;
        for (column, c) in line.chars().enumerate() {
            let Some(rows) = glyph(c) else { continue };
            let left = x + (PAD + column as i32 * ADVANCE) * scale;
            for (row, bits) in rows.iter().enumerate() {
                let mut cell = 0;
                while cell < WIDTH {
                    if bits >> (WIDTH - 1 - cell) & 1 == 0 {
                        cell += 1;
                        continue;
                    }
                    let start = cell;
                    while cell < WIDTH && bits >> (WIDTH - 1 - cell) & 1 == 1 {
                        cell += 1;
                    }
                    let r = Rect {
                        left: left + start * scale,
                        top: top + row as i32 * scale,
                        right: left + cell * scale,
                        bottom: top + (row as i32 + 1) * scale,
                    };
                    lit.extend(clip(r));
                }
            }
        }
    }
    Some((backing, lit))
}

/// The overlay's lines for a second's stats: frames a second and ms a
/// frame, then ms of a frame preparing on the CPU and waiting for the GPU.
pub(crate) fn lines(fps: f64, ms: f64, cpu: f64, gpu: f64) -> [String; 2] {
    [format!("{fps:.1} fps {ms:.1} ms"), format!("cpu {cpu:.1} gpu {gpu:.1} ms")]
}

#[cfg(test)]
mod tests {
    use super::*;

    const BIG: i32 = 10_000;

    #[test]
    fn one_cell_glyph_rows_become_runs() {
        // '.': two rows of one run two cells wide, cells 1 and 2.
        let (backing, lit) = layout(&["."], 10, 20, 3, BIG, BIG).unwrap();
        let top = 20 + PAD * 3;
        let left = 10 + PAD * 3;
        assert_eq!(
            lit,
            vec![
                Rect { left: left + 3, top: top + 15, right: left + 9, bottom: top + 18 },
                Rect { left: left + 3, top: top + 18, right: left + 9, bottom: top + 21 },
            ]
        );
        assert_eq!(backing, Rect { left: 10, top: 20, right: 10 + (2 * PAD + 5) * 3, bottom: 20 + (2 * PAD + 7) * 3 });
    }

    #[test]
    fn runs_split_on_gaps() {
        // '0' row 2 is 10011: a run of one, then a run of two.
        let (_, lit) = layout(&["0"], 0, 0, 1, BIG, BIG).unwrap();
        let row2: Vec<_> = lit.iter().filter(|r| r.top == PAD + 2).collect();
        assert_eq!(row2.len(), 2);
        assert_eq!((row2[0].left, row2[0].right), (PAD, PAD + 1));
        assert_eq!((row2[1].left, row2[1].right), (PAD + 3, PAD + 5));
    }

    #[test]
    fn every_cell_of_every_glyph_is_inside_its_box() {
        let text = "0123456789.- cfgmpsu";
        let (backing, lit) = layout(&[text, "x"], 0, 0, 2, BIG, BIG).unwrap();
        assert!(!lit.is_empty());
        for r in &lit {
            assert!(r.left >= backing.left && r.right <= backing.right, "{r:?}");
            assert!(r.top >= backing.top && r.bottom <= backing.bottom, "{r:?}");
            assert!(r.left < r.right && r.top < r.bottom);
            assert_eq!((r.right - r.left) % 2, 0);
            assert_eq!(r.bottom - r.top, 2);
        }
        // The number of lit cells matches the font's bits.
        let cells: i32 = lit.iter().map(|r| (r.right - r.left) / 2).sum();
        let bits: u32 = text.chars().filter_map(glyph).flatten().map(u8::count_ones).sum();
        assert_eq!(cells as u32, bits);
    }

    #[test]
    fn second_line_and_columns_advance() {
        let (_, lit) = layout(&["1", " 1"], 0, 0, 1, BIG, BIG).unwrap();
        // The top row of '1' (00100) on each line.
        assert!(lit.iter().any(|r| r.top == PAD && r.left == PAD + 2));
        assert!(lit.iter().any(|r| r.top == PAD + LINE && r.left == PAD + ADVANCE + 2));
        assert_eq!(lit.len(), 2 * 7);
    }

    #[test]
    fn clipped_to_the_target() {
        let (backing, lit) = layout(&["8888"], 0, 0, 4, 30, 20).unwrap();
        assert_eq!((backing.right, backing.bottom), (30, 20));
        assert!(lit.iter().all(|r| r.right <= 30 && r.bottom <= 20 && r.left < r.right && r.top < r.bottom));
        assert!(layout(&["8"], 50, 50, 4, 30, 20).is_none());
        assert!(layout(&[], 0, 0, 4, 30, 20).is_none());
    }

    #[test]
    fn scale_and_lines() {
        assert_eq!(scale(1440), 4);
        assert_eq!(scale(1080), 3);
        assert_eq!(scale(480), 2);
        assert_eq!(lines(29.96, 33.37, 1.04, 20.0), ["30.0 fps 33.4 ms".to_string(), "cpu 1.0 gpu 20.0 ms".to_string()]);
    }
}
