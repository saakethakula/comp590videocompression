use std::env;
use std::path::PathBuf;

use ffmpeg_sidecar::command::FfmpegCommand;
use workspace_root::get_workspace_root;

use std::fs::File;
use std::io::BufReader;
use std::io::{BufWriter, Write};

use bitbit::BitReader;
use bitbit::BitWriter;
use bitbit::MSB;

use toy_ac::decoder::Decoder;
use toy_ac::encoder::Encoder;
use toy_ac::symbol_model::SymbolModel;

use ffmpeg_sidecar::event::StreamTypeSpecificData::Video;

#[derive(Clone)]
struct AdaptiveByteModel {
    symbols: [u8; 256],
    counts: [u32; 256],
    total: u32,
}

impl AdaptiveByteModel {
    fn new() -> Self {
        Self {
            symbols: std::array::from_fn(|idx| idx as u8),
            counts: [1; 256],
            total: 256,
        }
    }

    fn incr_count(&mut self, symbol: u8) {
        self.counts[symbol as usize] += 1;
        self.total += 1;
        self.normalize();
    }

    fn normalize(&mut self) {
        while self.total >= 1_000_000 {
            let mut new_total = 0;
            for count in &mut self.counts {
                *count = if *count < 3 { 1 } else { *count / 2 };
                new_total += *count;
            }
            self.total = new_total;
        }
    }
}

impl SymbolModel<u8> for AdaptiveByteModel {
    fn contains(&self, _s: &u8) -> bool {
        true
    }

    fn total(&self) -> u32 {
        self.total
    }

    fn interval(&self, s: &u8) -> (u32, u32) {
        let idx = *s as usize;
        let mut sum = 0;
        for i in 0..idx {
            sum += self.counts[i];
        }
        (sum, sum + self.counts[idx])
    }

    fn lookup(&self, v: u32) -> (&u8, u32, u32) {
        if v >= self.total {
            panic!("Lookup value out of range");
        }

        let mut sum = 0;
        for idx in 0..256 {
            let next = sum + self.counts[idx];
            if v < next {
                return (&self.symbols[idx], sum, next);
            }
            sum = next;
        }
        panic!("Lookup failed");
    }
}

fn clamped_gradient_predictor(left: u8, up: u8, up_left: u8) -> u8 {
    let value = left as i16 + up as i16 - up_left as i16;
    value.clamp(0, 255) as u8
}

fn predict_pixel(prior_frame: &[u8], decoded_frame: &[u8], width: usize, row: usize, col: usize) -> u8 {
    let pixel_index = row * width + col;
    let prev = prior_frame[pixel_index];
    let left = if col > 0 {
        decoded_frame[pixel_index - 1]
    } else {
        prev
    };
    let up = if row > 0 {
        decoded_frame[pixel_index - width]
    } else {
        prev
    };
    let up_left = if row > 0 && col > 0 {
        decoded_frame[pixel_index - width - 1]
    } else {
        prev
    };
    let spatial = clamped_gradient_predictor(left, up, up_left);
    ((spatial as u16 + prev as u16 + 1) / 2) as u8
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Make sure ffmpeg is installed
    ffmpeg_sidecar::download::auto_download().unwrap();

    // Command line options
    // -verbose, -no_verbose                Default: -no_verbose
    // -report, -no_report                  Default: -report
    // -check_decode, -no_check_decode      Default: -no_check_decode
    // -skip_count n                        Default: -skip_count 0
    // -count n                             Default: -count 10
    // -in file_path                        Default: bourne.mp4 in data subdirectory of workplace
    // -out file_path                       Default: out.dat in data subdirectory of workplace

    // Set up default values of options
    let mut verbose = false;
    let mut report = true;
    let mut check_decode = false;
    let mut skip_count = 0;
    let mut count = 10;

    let mut data_folder_path = get_workspace_root();
    data_folder_path.push("data");

    let mut input_file_path = data_folder_path.join("bourne.mp4");
    let mut output_file_path = data_folder_path.join("out.dat");

    parse_args(
        &mut verbose,
        &mut report,
        &mut check_decode,
        &mut skip_count,
        &mut count,
        &mut input_file_path,
        &mut output_file_path,
    );

    // Run an FFmpeg command to decode video from inptu_file_path
    // Get output as grayscale (i.e., just the Y plane)

    let mut iter = FfmpegCommand::new() // <- Builder API like `std::process::Command`
        .input(input_file_path.to_str().unwrap())
        .format("rawvideo")
        .pix_fmt("gray8")
        .output("-")
        .spawn()? // <- Ordinary `std::process::Child`
        .iter()?; // <- Blocking iterator over logs and output

    // Figure out geometry of frame.
    let mut width = 0;
    let mut height = 0;

    let metadata = iter.collect_metadata()?;
    for i in 0..metadata.output_streams.len() {
        match &metadata.output_streams[i].type_specific_data {
            Video(vid_stream) => {
                width = vid_stream.width;
                height = vid_stream.height;

                if verbose {
                    println!(
                        "Found video stream at output stream index {} with dimensions {} x {}",
                        i, width, height
                    );
                }
                break;
            }
            _ => (),
        }
    }
    assert!(width != 0);
    assert!(height != 0);

    // Set up initial prior frame as uniform medium gray (y = 128)
    let mut prior_frame = vec![128 as u8; (width * height) as usize];

    let output_file = match File::create(&output_file_path) {
        Err(_) => panic!("Error opening output file"),
        Ok(f) => f,
    };

    // Setup bit writer and arithmetic encoder.

    let mut buf_writer = BufWriter::new(output_file);
    let mut bw = BitWriter::new(&mut buf_writer);

    let mut enc = Encoder::new();

    // Set up arithmetic coding context(s)
    let mut context_models = vec![AdaptiveByteModel::new(); 256];
    let mut encoded_frames = 0_u32;

    // Process frames
    for frame in iter.filter_frames() {
        if frame.frame_num < skip_count {
            if verbose {
                println!("Skipping frame {}", frame.frame_num);
            }
        } else if frame.frame_num < skip_count + count {
            let current_frame: Vec<u8> = frame.data; // <- raw pixel y values
            let mut reconstructed_frame = vec![0_u8; current_frame.len()];

            let bits_written_at_start = enc.bits_written();

            // Process pixels in row major order.
            for r in 0..height as usize {
                for c in 0..width as usize {
                    let pixel_index = r * width as usize + c;
                    let predictor =
                        predict_pixel(&prior_frame, &reconstructed_frame, width as usize, r, c);
                    let context = predictor as usize;

                    let residual = current_frame[pixel_index].wrapping_sub(predictor);

                    enc.encode(&residual, &context_models[context], &mut bw);

                    context_models[context].incr_count(residual);
                    reconstructed_frame[pixel_index] = predictor.wrapping_add(residual);
                }
            }

            prior_frame = current_frame;
            encoded_frames += 1;

            let bits_written_at_end = enc.bits_written();

            if verbose {
                println!(
                    "frame: {}, compressed size (bits): {}",
                    frame.frame_num,
                    bits_written_at_end - bits_written_at_start
                );
            }
        } else {
            break;
        }
    }

    // Tie off arithmetic encoder and flush to file.
    enc.finish(&mut bw)?;
    bw.pad_to_byte()?;
    buf_writer.flush()?;

    // Decompress and check for correctness.
    if check_decode {
        let output_file = match File::open(&output_file_path) {
            Err(_) => panic!("Error opening output file"),
            Ok(f) => f,
        };
        let mut buf_reader = BufReader::new(output_file);
        let mut br: BitReader<_, MSB> = BitReader::new(&mut buf_reader);

        let iter = FfmpegCommand::new() // <- Builder API like `std::process::Command`
            .input(input_file_path.to_str().unwrap())
            .format("rawvideo")
            .pix_fmt("gray8")
            .output("-")
            .spawn()? // <- Ordinary `std::process::Child`
            .iter()?; // <- Blocking iterator over logs and output

        let mut dec = Decoder::new();

        let mut context_models = vec![AdaptiveByteModel::new(); 256];

        // Set up initial prior frame as uniform medium gray
        let mut prior_frame = vec![128 as u8; (width * height) as usize];

        'outer_loop: 
        for frame in iter.filter_frames() {
            if frame.frame_num < skip_count {
                continue;
            } else if frame.frame_num < skip_count + encoded_frames {
                if verbose {
                    print!("Checking frame: {} ... ", frame.frame_num);
                }

                let current_frame: Vec<u8> = frame.data; // <- raw pixel y values
                let mut reconstructed_frame = vec![0_u8; current_frame.len()];

                // Process pixels in row major order.
                for r in 0..height as usize {
                    for c in 0..width as usize {
                        let pixel_index = r * width as usize + c;
                        let predictor =
                            predict_pixel(&prior_frame, &reconstructed_frame, width as usize, r, c);
                        let context = predictor as usize;
                        let residual = *dec.decode(&context_models[context], &mut br);
                        context_models[context].incr_count(residual);

                        let pixel_value = predictor.wrapping_add(residual);
                        reconstructed_frame[pixel_index] = pixel_value;

                        if pixel_value != current_frame[pixel_index] {
                            println!(
                                " error at ({}, {}), should decode {}, got {}",
                                c, r, current_frame[pixel_index], pixel_value
                            );
                            println!("Abandoning check of remaining frames");
                            break 'outer_loop;
                        }
                    }
                }
                println!("correct.");
                prior_frame = current_frame;
            } else {
                break 'outer_loop;
            }
        }
    }

    // Emit report
    if report {
        if encoded_frames == 0 {
            println!("0 frames encoded");
        } else {
            println!(
                "{} frames encoded, average size (bits): {}, compression ratio: {:.2}",
                encoded_frames,
                enc.bits_written() / encoded_frames as u64,
                (width * height * 8 * encoded_frames) as f64 / enc.bits_written() as f64
            )
        }
    }

    Ok(())
}

fn parse_args(
    verbose: &mut bool,
    report: &mut bool,
    check_decode: &mut bool,
    skip_count: &mut u32,
    count: &mut u32,
    input_file_path: &mut PathBuf,
    output_file_path: &mut PathBuf,
) -> () {
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        if arg == "-verbose" {
            *verbose = true;
        } else if arg == "-no_verbose" {
            *verbose = false;
        } else if arg == "-report" {
            *report = true;
        } else if arg == "-no_report" {
            *report = false;
        } else if arg == "-check_decode" {
            *check_decode = true;
        } else if arg == "-no_check_decode" {
            *check_decode = false;
        } else if arg == "-skip_count" {
            match args.next() {
                Some(skip_count_string) => {
                    *skip_count = skip_count_string.parse::<u32>().unwrap();
                }
                None => {
                    panic!("Expected count after -skip_count option");
                }
            }
        } else if arg == "-count" {
            match args.next() {
                Some(count_string) => {
                    *count = count_string.parse::<u32>().unwrap();
                }
                None => {
                    panic!("Expected count after -count option");
                }
            }
        } else if arg == "-in" {
            match args.next() {
                Some(input_file_path_string) => {
                    *input_file_path = PathBuf::from(input_file_path_string);
                }
                None => {
                    panic!("Expected input file name after -in option");
                }
            }
        } else if arg == "-out" {
            match args.next() {
                Some(output_file_path_string) => {
                    *output_file_path = PathBuf::from(output_file_path_string);
                }
                None => {
                    panic!("Expected output file name after -out option");
                }
            }
        }
    }
}
