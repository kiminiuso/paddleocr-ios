//
//  Utils.m
//  CardRecognize
//
//  Created by kiminiuso on R 3/06/17.
//

#import "Utils.h"

@implementation Utils

+ (NSArray *)readLabelsFromFile:(NSString *)labelFilePath {

    NSString *content = [NSString stringWithContentsOfFile:labelFilePath encoding:NSUTF8StringEncoding error:nil];
    NSArray *lines = [content componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
    NSMutableArray *ret = [[NSMutableArray alloc] init];
    for (int i = 0; i < lines.count; ++i) {
        [ret addObject:@""];
    }
    NSUInteger cnt = 0;
    for (id line in lines) {
        NSString *l = [(NSString *) line stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        if ([l length] == 0)
            continue;
        NSArray *segs = [l componentsSeparatedByString:@":"];
        NSUInteger key;
        NSString *value;
        if ([segs count] != 2) {
            key = cnt;
            value = [segs[0] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        } else {
            key = [[segs[0] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] integerValue];
            value = [segs[1] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        }

        ret[key] = value;
        cnt += 1;
    }
    return [NSArray arrayWithArray:ret];
}

@end
