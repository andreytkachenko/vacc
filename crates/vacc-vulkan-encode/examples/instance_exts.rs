//! List instance extensions exposed by the Vulkan loader (headless debug).
fn main() {
    let entry = unsafe { ash::Entry::load() }.unwrap();
    for ext in unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap() {
        let name: String = ext
            .extension_name
            .iter()
            .take_while(|&&b| b != 0)
            .map(|b| *b as u8 as char)
            .collect();
        println!("  {name} v{}", ext.spec_version);
    }
}
