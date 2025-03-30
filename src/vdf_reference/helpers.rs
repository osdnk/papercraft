
macro_rules! println_with_timestamp {
    ($($arg:tt)*) => {
        // Get the current local time
        let now = chrono::Local::now();
        // Format the time however you like. Here we use "%Y-%m-%d %H:%M:%S"
        print!("[{}] ", now.format("%Y-%m-%d %H:%M:%S"));
        println!($($arg)*);
    };
}

pub(crate) use println_with_timestamp;
