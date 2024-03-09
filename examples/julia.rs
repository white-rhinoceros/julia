extern crate julia;
extern crate image;

use std::fs::File;
use image::codecs::png::PngEncoder;
use image::{ColorType, ImageEncoder};
use julia::gen_julia;

// Количество байт на пиксель (Rgba8).
const BIT_PER_PIXEL: usize = 4;

fn main() {
    println!("Генерируем фрактал Julia!");

    // Размер фрактала.
    let bounds = (10000, 10000, BIT_PER_PIXEL);

    // Наше изображение.
    let mut pixels: Vec<u8> = vec![0; bounds.0 * bounds.1 * bounds.2];

    gen_julia(&mut pixels, bounds);

    write_image("./result/julia.png", &pixels, (bounds.0, bounds.1))
        .expect("Ошибка при записи PNG файла");
}

/// Записывает буфер `pixel`, размеры которого заданы аргументом `bounds`,
/// в файл с именем `filename`.
fn write_image(filename: &str, pixels: &[u8], bounds: (usize, usize)) -> Result<(), String> {
     match File::create(filename) {
         Ok(output) => {
             let encoder = PngEncoder::new(output);
             let result = encoder
                 .write_image(&pixels, bounds.0 as u32, bounds.1 as u32, ColorType::Rgba8);
             if let Err(err) = result {
                 return Err(err.to_string());
             }
         }
         Err(err) => {
             return Err(err.to_string());
         }
     }

    Ok(())
}