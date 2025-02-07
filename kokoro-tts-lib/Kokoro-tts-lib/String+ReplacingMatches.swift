//
//  kokoro-tts-lib
//
import Foundation

extension String {
    /// Replace all regex matches using a closure.
    func replacingMatches(pattern: String,
                           options: NSRegularExpression.Options = [],
                           using transform: (NSTextCheckingResult, String) -> String) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: options) else { return self }
        let nsSelf = self as NSString
        let matches = regex.matches(in: self, options: [], range: NSRange(location: 0, length: nsSelf.length))
        var result = self
        // Replace matches in reverse order so that earlier replacements don’t change later ranges.
        for match in matches.reversed() {
            let range = match.range
            let matchedString = nsSelf.substring(with: range)
            let replacement = transform(match, matchedString)
            result = (result as NSString).replacingCharacters(in: range, with: replacement)
        }
        return result
    }
    
    /// Replace all regex matches with a fixed template.
    func replacingMatches(pattern: String,
                           options: NSRegularExpression.Options = [],
                           with template: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: options) else { return self }
        let range = NSRange(startIndex..., in: self)
        return regex.stringByReplacingMatches(in: self, options: [], range: range, withTemplate: template)
    }
}
